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
    GemmaForCausalLM,
    GenerationConfig,
)
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.trainer import _is_peft_model
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.trainer_utils import PredictionOutput, EvalLoopOutput
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from config.config import Config
from src.data.instruction import LPInstrucDataset
from src.model.model import GNNLLMOutput, CusCausalLMOutputWithPast
from src.data.types import InstrucInput, PretrainDatasetOutput
from src.trainer.dataloader import DataloaderMixin
from src.trainer.metric import metric_fn


from torchviz import make_dot

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

    def compute_loss_train(
        self, model, dataloader: List[InstrucInput]
    ) -> Tuple[torch.Tensor, GNNLLMOutput]:
        # 2. forward
        losses = []
        labels = []
        for batch in dataloader:
            batch = batch.to(self.accelerator.device)
            outputs: CusCausalLMOutputWithPast = model(
                input_ids=batch.input_ids, embeds=batch.embs, labels=batch.label_ids
            )
            # gf = make_dot(outputs.loss, params=dict(model.named_parameters()), show_attrs=True)
            if not model.training and outputs.labels is not None:
                labels.append(outputs.labels)

            losses.append(outputs.loss)

        if not model.training and len(labels) > 0:
            self.labels = torch.stack(labels)
        else:
            self.labels = None

        loss = torch.stack(losses).mean()
        return loss, outputs

    def compute_loss_test(
        self, model: GemmaForCausalLM, dataloader: List[InstrucInput]
    ) -> Tuple[torch.Tensor, GNNLLMOutput]:
        assert len(dataloader) == 1, "Only one batch is allowed in test mode"

        for batch in dataloader:
            batch = batch.to(self.accelerator.device)

            outputs: CausalLMOutputWithPast = model(
                input_ids=None, embeds=batch.embs, labels=batch.label_ids
            )

        dummy_loss = torch.tensor(0.0, device=outputs.logits.device)
        return dummy_loss, outputs

    def compute_loss_test_instruct_llm(
        self, model: GemmaForCausalLM, dataloader: List[InstrucInput]
    ) -> Tuple[torch.Tensor, GNNLLMOutput]:
        assert len(dataloader) == 1, "Only one batch is allowed in test mode"

        for batch in dataloader:
            batch = batch.to(self.accelerator.device)
            # 修剪 input_ids: 去除 label_ids
            for i in range(batch.label_ids.shape[1] - 1, -1, -1):
                if batch.label_ids[0][i] == -100:
                    break
            batch.input_ids = batch.input_ids[:, : i + 1]

            outputs: GenerateDecoderOnlyOutput = model.generate(
                inputs=batch.input_ids,
                generation_config=GenerationConfig(max_new_tokens=70, min_new_tokens=2),
                embeds=batch.embs,
                return_dict_in_generate=True,
            )

        # outputs
        #       sequences : batch_size * tokens
        # 1. 修剪成 2048这样正确的长度 -> batch.label_ids.shape[1, 2048]
        # 2. ret_object.logits = sequences_fixed
        outputs.logits = outputs.sequences[:, : batch.label_ids.shape[1]]
        # outputs["logits"] = outputs.logits
        # del outputs["sequences"]
        # 这里只是为了仅保留一个 kv
        outputs["sequences"] = outputs.logits

        # loss
        #   dummy loss -> torch.tensor(0, device=)
        dummy_loss = torch.tensor(0.0, device=outputs.logits.device)

        return dummy_loss, outputs

    def compute_loss(self, model, inputs: PretrainDatasetOutput, return_outputs=False):
        """
        Perform model.forward and calculate the loss.
        """
        # >>> encode <<<
        outputs: GNNLLMOutput = model(data=inputs)

        # >>> decode <<<
        # 1. create dataloader for instruct tuning
        dataloader: list[InstrucInput] = self.get_instruct_dataloader(inputs, outputs)

        if self._stage == "train":
            loss, outputs = self.compute_loss_train(model, dataloader)
        elif self._stage == "eval":
            loss, outputs = self.compute_loss_train(model, dataloader)
        elif self._stage == "test":
            loss, outputs = self.compute_loss_test(model, dataloader)
        else:
            raise ValueError(f"Invalid stage: {self._stage}")

        self.log({f"{self._stage}_loss": loss.detach().item()})

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs) -> torch.Tensor:
        """
        invoke the compute_loss method, then backward loss and return the loss tensor.
        :return: The loss tensor.
        """
        self._stage = "train"
        return super().training_step(model, inputs)

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial=None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        self._stage = "train"
        return super().train(
            resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs
        )

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        loss, logits, labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=['past_key_values', 'labels'])
        if self.labels is not None:
            labels = self.labels
        return loss, logits, labels


    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: List[str] | None = None,
        metric_key_prefix: str = "test",
    ) -> PredictionOutput:
        self._stage = "test"
        return super().predict(test_dataset, ignore_keys, metric_key_prefix)

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        self._stage = "eval"
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
