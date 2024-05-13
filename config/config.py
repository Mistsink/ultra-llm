from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from transformers import TrainingArguments


@dataclass
class ModelDetailConfig:
    class_name: str = field(default="GemmaForCausalLM")
    message_func: str = field(default="message_passing")
    aggregate_func: str = field(default="sum")
    short_cut: bool = field(default=True)
    layer_norm: bool = field(default=True)
    input_dim: int = field(default=768)
    hidden_dims: List[int] = field(default_factory=lambda: [768])

@dataclass
class ModelConfig:
    class_name: str = field(default="GemmaForCausalLM")
    llm_name: str = field(default="gpt2")
    cache_dir: str = field(default="")
    hf_token: str = field(default="<g-exchange-info-head>")
    relation_model: ModelDetailConfig = field(default_factory=ModelDetailConfig)
    entity_model: ModelDetailConfig = field(default_factory=ModelDetailConfig)

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self.relation_model = ModelDetailConfig(**kwargs.get("relation_model", {}))
        self.entity_model = ModelDetailConfig(**kwargs.get("entity_model", {}))


@dataclass
class DatasetConfig:
    class_name: str = field(default="KGLLMData")
    root: str = field(default="")
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskConfig:
    name: str = field(default="KGLLM")
    strict_negative: bool = field(default=True)
    adversarial_temperature: float = field(default=1.0)
    metric: List[str] = field(default_factory=lambda: ["loss"])
    num_negative: int = field(default=50)
    instruct_len: int = field(default=512)
    prompt_len: int = field(default=1024 * 8)
    num_mask: int = field(default=64)
    num_neighbors: list[int] = field(default_factory=lambda: [-1, 50])


@dataclass
class OptimizerConfig:
    class_name: str = field(default="AdamW")
    lr: float = field(default=5e-5)

@dataclass
class TrainConfig(TrainingArguments):
    # rewrite inner default config
    per_device_train_batch_size: Optional[int] = field(default=None)

    output_dir: str = field(default="output")
    batch_size: int = field(default=8)
    instruct_batch_size: int = field(default=1)
    batch_per_epoch: int = field(default=1000)

    log_interval: int = field(default=100)
    num_train_epochs: int = field(default=10)
    fast_test: int = field(default=-1)

    evaluation_strategy: str = field(default="epoch")
    save_strategy: str = field(default="epoch") # steps / epoch
    save_steps: int = field(default=1000)
    metric_for_best_model: Optional[str] = field(
        default=None, metadata={"help": "The metric to use to compare two different models."}
    )
    greater_is_better: Optional[bool] = field(
        default=None, metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."}
    )


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self.train = TrainConfig(**kwargs.get("train", {}))
        self.model = ModelConfig(**kwargs.get("model", {}))
        self.dataset = DatasetConfig(**kwargs.get("dataset", {}))
        self.task = TaskConfig(**kwargs.get("task", {}))
        self.optimizer = OptimizerConfig(**kwargs.get("optimizer", {}))

