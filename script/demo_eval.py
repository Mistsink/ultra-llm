import os

os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ["HF_HUB_OFFLINE"] = "0"
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


def parse_args(config_path: str) -> Config:
    parser = HfArgumentParser(Config)
    cfg: Config = parser.parse_yaml_file(config_path)[0]
    cfg.train = cfg.train.set_dataloader(
        train_batch_size=cfg.train.batch_size, eval_batch_size=cfg.train.batch_size
    )

    # get_logger().
    return cfg


def get_data(cfg: Config) -> tuple[InMemoryDataset, CustomData, CustomData, CustomData]:
    dataset = util.build_dataset(cfg)
    return dataset, dataset[0], dataset[1], dataset[2]


if __name__ == "__main__":

    cfg = parse_args("config/pretrain/eval_llm_match.yaml")
    transformers.set_seed(cfg.train.seed)

    task_name = cfg.task.name

    # data sampler, loader, collator -> custom trainer
    dataset, train_data, valid_data, test_data = get_data(cfg=cfg)

    tokenizer, model = build_tokenizer_model(cfg)

    trainer = KGLLMTrainer(
        cfg=cfg,
        model=model,
        args=cfg.train,
        train_dataset=train_data,
        eval_dataset=valid_data,
        tokenizer=tokenizer,
        compute_metrics=ROUGE(cfg, tokenizer),
    )

    # trainer.train()
    line = "-----" * 10
    # metrics = trainer.predict(test_dataset=valid_data)
    # print(line)
    # print(f"predict on VALID_data")
    # print(metrics)
    # print(line)
    metrics = trainer.predict(test_dataset=test_data)
    print(line)
    print(f"predict on TEST_data")
    print(metrics)
    print(line)
