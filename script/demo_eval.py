import os
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ['HF_HUB_OFFLINE'] = "0"
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from torch_geometric.data import Data, InMemoryDataset
import transformers
from transformers import Trainer, TrainingArguments, HfArgumentParser

from script.build_model import build_model, build_tokenizer_model
from src.trainer.metric import ROUGE, metric_fn
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


def parse_args(config_path: str) -> Config:
    parser = HfArgumentParser(Config)
    cfg: Config = parser.parse_yaml_file(config_path)[0]
    cfg.train = cfg.train.set_dataloader(train_batch_size=cfg.train.batch_size, eval_batch_size=cfg.train.batch_size)

    # get_logger().
    return cfg


def get_data(cfg: Config) -> tuple[InMemoryDataset, CustomData, CustomData, CustomData]:
    dataset = util.build_dataset(cfg)
    return dataset, dataset[0], dataset[1], dataset[2]


if __name__ == "__main__":

    cfg = parse_args("config/pretrain/eval.yaml")
    transformers.set_seed(cfg.train.seed)

    task_name = cfg.task.name

    # data sampler, loader, collator -> custom trainer
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
        compute_metrics=ROUGE(cfg, tokenizer),
        # callbacks=[VLogCallback(save_path=os.path.join(cfg.output_dir, "history.png"))],
    )
    # trainer.train()
    metrics = trainer.evaluate(eval_dataset=valid_data)
    print(metrics)

    # if trainer.is_deepspeed_enabled:
    #     trainer.deepspeed = trainer.model_wrapped
    trainer.save_model()


    # metrics = trainer.evaluate(eval_dataset=valid_data)

    # trainer.train(resume_from_checkpoint="/disk1/hy/ultra_llm/output/checkpoint-4000/")
    # trainer._load_from_checkpoint("/disk1/hy/ultra_llm/output/checkpoint-4000/")
    metrics = trainer.evaluate(eval_dataset=valid_data)
    print(metrics)
    metrics = trainer.evaluate(eval_dataset=test_data)
    print(metrics)

    # trainer.ev

    # for transductive setting, use the whole graph for filtered ranking
    # filtered_data = [
    #     Data(
    #         edge_index=torch.cat(
    #             [
    #                 trg.target_edge_index,
    #                 valg.target_edge_index,
    #                 testg.target_edge_index,
    #             ],
    #             dim=1,
    #         ),
    #         edge_type=torch.cat(
    #             [
    #                 trg.target_edge_type,
    #                 valg.target_edge_type,
    #                 testg.target_edge_type,
    #             ]
    #         ),
    #         num_nodes=trg.num_nodes,
    #     )
    #     for trg, valg, testg in zip(train_data, valid_data, test_data)
    # ]
