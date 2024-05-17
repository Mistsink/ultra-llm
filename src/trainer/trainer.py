from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    EvalPrediction,
    PreTrainedTokenizerBase,
    TrainerCallback,
    DataCollator,
)
from transformers.trainer import _is_peft_model
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.trainer_utils import PredictionOutput, EvalLoopOutput
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from config.config import Config
from src.data.instruction import LPInstrucDataset
from src.model.model import GNNLLMOutput
from src.data.types import InstrucInput, PretrainDatasetOutput
from src.trainer.dataloader import DataloaderMixin
from src.trainer.metric import metric_fn


class KGLLMTrainer(DataloaderMixin, Trainer):
    def __init__(
        self,
        cfg: Config,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
    ):
        if args is None:
            args = cfg.train
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics if compute_metrics is not None else metric_fn,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        self.cfg = cfg

    def compute_loss_train(self, model, inputs: PretrainDatasetOutput, return_outputs):
        # >>> encode <<<
        outputs: GNNLLMOutput = model(data=inputs)

        # >>> decode <<<
        # 1. create dataloader for instruct tuning
        dataloader: list[InstrucInput] = self.get_instruct_dataloader(inputs, outputs)

        # 2. forward
        losses = []
        for batch in dataloader:
            batch = batch.to(self.accelerator.device)
            outputs: CausalLMOutputWithPast = model(
                input_ids=batch.input_ids, embeds=batch.embs, labels=batch.label_ids
            )

            losses.append(outputs.loss)

        loss = torch.stack(losses).mean()

        self.log({
            "train_loss": loss.detach().item()
        })

        return (loss, outputs) if return_outputs else loss

    def compute_loss(self, model, inputs: PretrainDatasetOutput, return_outputs=False):
        """
        Perform model.forward and calculate the loss.
        """
        # >>> encode <<<
        outputs: GNNLLMOutput = model(data=inputs)

        # >>> decode <<<
        # 1. create dataloader for instruct tuning
        dataloader: list[InstrucInput] = self.get_instruct_dataloader(inputs, outputs)

        # 2. forward
        losses = []
        for batch in dataloader:
            batch = batch.to(self.accelerator.device)
            outputs: CausalLMOutputWithPast = model(
                input_ids=batch.input_ids, embeds=batch.embs, labels=batch.label_ids
            )

            losses.append(outputs.loss)

        loss = torch.stack(losses).mean()

        self.log({
            "train_loss": loss.detach().item()
        })

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs) -> torch.Tensor:
        """
        invoke the compute_loss method, then backward loss and return the loss tensor.
        :return: The loss tensor.
        """
        self._stage = 'train'
        return super().training_step(model, inputs)
    
    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial= None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        self._stage = 'train'
        return super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: List[str] | None = None,
        metric_key_prefix: str = "test",
    ) -> PredictionOutput:
        self._stage = 'test'
        return super().predict(test_dataset, ignore_keys, metric_key_prefix)
    

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        self._stage = 'eval'
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)


if __name__ == "__main__":
    from torchkeras.tools.transformers import VLogCallback

    trainer = KGLLMTrainer(callbacks=[VLogCallback()])
    trainer.get_eval_dataloader()
    trainer.get_train_dataloader()
    trainer.train()
