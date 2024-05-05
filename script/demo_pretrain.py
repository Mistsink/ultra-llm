import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import copy
import math
import pprint
from itertools import islice
from functools import partial

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data, InMemoryDataset
import transformers
from transformers import Trainer, TrainingArguments, HfArgumentParser
import accelerate
from torchkeras.tools.transformers import VLogCallback

from script.build_model import build_model, build_tokenizer_model
from src.trainer.metric import metric_fn
from src.trainer.trainer import KGLLMTrainer
from config.config import Config
from src.data.datasets import FB15k237Inductive
from src.data.types import CustomData
from src.ultra import tasks, util
from src.ultra.models import Ultra


separator = ">" * 30
line = "-" * 30


def multigraph_collator(batch, train_graphs):
    num_graphs = len(train_graphs)
    probs = torch.tensor([graph.edge_index.shape[1] for graph in train_graphs]).float()
    probs /= probs.sum()
    graph_id = torch.multinomial(probs, 1, replacement=False).item()

    graph = train_graphs[graph_id]
    bs = len(batch)
    edge_mask = torch.randperm(graph.target_edge_index.shape[1])[:bs]

    batch = torch.cat(
        [
            graph.target_edge_index[:, edge_mask],
            graph.target_edge_type[edge_mask].unsqueeze(0),
        ]
    ).t()
    return graph, batch


# here we assume that train_data and valid_data are tuples of datasets
def train_and_validate(
    cfg, model, train_data, valid_data, filtered_data=None, batch_per_epoch=None
):

    if cfg.train.num_epoch == 0:
        return

    world_size = util.get_world_size()
    rank = util.get_rank()

    train_triplets = torch.cat(
        [
            torch.cat([g.target_edge_index, g.target_edge_type.unsqueeze(0)]).t()
            for g in train_data
        ]
    )
    sampler = torch_data.DistributedSampler(train_triplets, world_size, rank)
    train_loader = torch_data.DataLoader(
        train_triplets,
        cfg.train.batch_size,
        sampler=sampler,
        collate_fn=partial(multigraph_collator, train_graphs=train_data),
    )

    batch_per_epoch = batch_per_epoch or len(train_loader)

    cls = cfg.optimizer.pop("class")
    optimizer = getattr(optim, cls)(model.parameters(), **cfg.optimizer)
    num_params = sum(p.numel() for p in model.parameters())
    logger.warning(line)
    logger.warning(f"Number of parameters: {num_params}")

    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        parallel_model = model

    step = math.ceil(cfg.train.num_epoch / 10)
    best_result = float("-inf")
    best_epoch = -1

    batch_id = 0
    for i in range(0, cfg.train.num_epoch, step):
        parallel_model.train()
        for epoch in range(i, min(cfg.train.num_epoch, i + step)):
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning("Epoch %d begin" % epoch)

            losses = []
            sampler.set_epoch(epoch)
            for batch in islice(train_loader, batch_per_epoch):
                # now at each step we sample a new graph and edges from it
                train_graph, batch = batch
                batch = tasks.negative_sampling(
                    train_graph,
                    batch,
                    cfg.task.num_negative,
                    strict=cfg.task.strict_negative,
                )
                pred = parallel_model(train_graph, batch)
                target = torch.zeros_like(pred)
                target[:, 0] = 1
                loss = F.binary_cross_entropy_with_logits(
                    pred, target, reduction="none"
                )
                neg_weight = torch.ones_like(pred)
                if cfg.task.adversarial_temperature > 0:
                    with torch.no_grad():
                        neg_weight[:, 1:] = F.softmax(
                            pred[:, 1:] / cfg.task.adversarial_temperature, dim=-1
                        )
                else:
                    neg_weight[:, 1:] = 1 / cfg.task.num_negative
                loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
                loss = loss.mean()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if util.get_rank() == 0 and batch_id % cfg.train.log_interval == 0:
                    logger.warning(separator)
                    logger.warning("binary cross entropy: %g" % loss)
                losses.append(loss.item())
                batch_id += 1

            if util.get_rank() == 0:
                avg_loss = sum(losses) / len(losses)
                logger.warning(separator)
                logger.warning("Epoch %d end" % epoch)
                logger.warning(line)
                logger.warning("average binary cross entropy: %g" % avg_loss)

        epoch = min(cfg.train.num_epoch, i + step)
        if rank == 0:
            logger.warning("Save checkpoint to model_epoch_%d.pth" % epoch)
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
            torch.save(state, "model_epoch_%d.pth" % epoch)
        util.synchronize()

        if rank == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")
        result = test(cfg, model, valid_data, filtered_data=filtered_data)
        if result > best_result:
            best_result = result
            best_epoch = epoch

    if rank == 0:
        logger.warning("Load checkpoint from model_epoch_%d.pth" % best_epoch)
    state = torch.load("model_epoch_%d.pth" % best_epoch, map_location=device)
    model.load_state_dict(state["model"])
    util.synchronize()


@torch.no_grad()
def test(cfg, model, test_data, filtered_data=None):
    world_size = util.get_world_size()
    rank = util.get_rank()

    # test_data is a tuple of validation/test datasets
    # process sequentially
    all_metrics = []
    for test_graph, filters in zip(test_data, filtered_data):

        test_triplets = torch.cat(
            [test_graph.target_edge_index, test_graph.target_edge_type.unsqueeze(0)]
        ).t()
        sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
        test_loader = torch_data.DataLoader(
            test_triplets, cfg.train.batch_size, sampler=sampler
        )

        model.eval()
        rankings = []
        num_negatives = []
        for batch in test_loader:
            t_batch, h_batch = tasks.all_negative(test_graph, batch)
            t_pred = model(test_graph, t_batch)
            h_pred = model(test_graph, h_batch)

            if filtered_data is None:
                t_mask, h_mask = tasks.strict_negative_mask(test_graph, batch)
            else:
                t_mask, h_mask = tasks.strict_negative_mask(filters, batch)
            pos_h_index, pos_t_index, pos_r_index = batch.t()
            t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
            h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
            num_t_negative = t_mask.sum(dim=-1)
            num_h_negative = h_mask.sum(dim=-1)

            rankings += [t_ranking, h_ranking]
            num_negatives += [num_t_negative, num_h_negative]

        ranking = torch.cat(rankings)
        num_negative = torch.cat(num_negatives)
        all_size = torch.zeros(world_size, dtype=torch.long, device=device)
        all_size[rank] = len(ranking)
        if world_size > 1:
            dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
        cum_size = all_size.cumsum(0)
        all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
        all_ranking[cum_size[rank] - all_size[rank] : cum_size[rank]] = ranking
        all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
        all_num_negative[cum_size[rank] - all_size[rank] : cum_size[rank]] = (
            num_negative
        )
        if world_size > 1:
            dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
            dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)

        if rank == 0:
            for metric in cfg.task.metric:
                if metric == "mr":
                    score = all_ranking.float().mean()
                elif metric == "mrr":
                    score = (1 / all_ranking.float()).mean()
                elif metric.startswith("hits@"):
                    values = metric[5:].split("_")
                    threshold = int(values[0])
                    if len(values) > 1:
                        num_sample = int(values[1])
                        # unbiased estimation
                        fp_rate = (all_ranking - 1).float() / all_num_negative
                        score = 0
                        for i in range(threshold):
                            # choose i false positive from num_sample - 1 negatives
                            num_comb = (
                                math.factorial(num_sample - 1)
                                / math.factorial(i)
                                / math.factorial(num_sample - i - 1)
                            )
                            score += (
                                num_comb
                                * (fp_rate**i)
                                * ((1 - fp_rate) ** (num_sample - i - 1))
                            )
                        score = score.mean()
                    else:
                        score = (all_ranking <= threshold).float().mean()
                logger.warning("%s: %g" % (metric, score))
        mrr = (1 / all_ranking.float()).mean()

        all_metrics.append(mrr)
        if rank == 0:
            logger.warning(separator)

    avg_metric = sum(all_metrics) / len(all_metrics)
    return avg_metric


def parse_args(config_path: str) -> Config:
    parser = HfArgumentParser(Config)
    return parser.parse_yaml_file(config_path)[0]


def get_data(cfg: Config) -> tuple[InMemoryDataset, CustomData, CustomData, CustomData]:
    dataset = util.build_dataset(cfg)
    return dataset, dataset[0], dataset[1], dataset[2]


if __name__ == "__main__":

    cfg = parse_args("config/pretrain/pretrain_0.yaml")
    transformers.set_seed(cfg.seed)

    task_name = cfg.task.name

    # TODO data sampler, loader, collator -> custom trainer
    dataset, train_data, valid_data, test_data = get_data(cfg=cfg)

    tokenizer, model = build_tokenizer_model(cfg)

    # assert (
    #     task_name == "MultiGraphPretraining"
    # ), "Only the MultiGraphPretraining task is allowed for this script"

    trainer = KGLLMTrainer(
        cfg=cfg,
        model=model,
        args=cfg.train,
        train_dataset=train_data,
        eval_dataset=valid_data,
        tokenizer=tokenizer,
        compute_metrics=metric_fn,
        # callbacks=[VLogCallback(save_path=os.path.join(cfg.output_dir, "history.png"))],
    )

    trainer.train()

    # for transductive setting, use the whole graph for filtered ranking
    filtered_data = [
        Data(
            edge_index=torch.cat(
                [
                    trg.target_edge_index,
                    valg.target_edge_index,
                    testg.target_edge_index,
                ],
                dim=1,
            ),
            edge_type=torch.cat(
                [
                    trg.target_edge_type,
                    valg.target_edge_type,
                    testg.target_edge_type,
                ]
            ),
            num_nodes=trg.num_nodes,
        )
        for trg, valg, testg in zip(train_data, valid_data, test_data)
    ]

    train_and_validate(
        cfg,
        model,
        train_data,
        valid_data if "fast_test" not in cfg.train else short_valid,
        filtered_data=filtered_data,
        batch_per_epoch=cfg.train.batch_per_epoch,
    )
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on valid")
    test(cfg, model, valid_data, filtered_data=filtered_data)
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on test")
    test(cfg, model, test_data, filtered_data=filtered_data)
