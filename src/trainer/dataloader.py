from pprint import pprint
from typing import TYPE_CHECKING, Optional
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer

from src.data.base_task.instruction import BaseTaskInstrucDataset
from src.data.llm_match.evaluate_llm_match import EvaluateLLMMatchEmbDataset
from src.data.llm_match.instruction_llm_match import LLMMatchInstrucDataset
from src.data.evaluate import EvaluateDataset
from src.data.instruction import LPInstrucDataset
from src.data.types import PretrainDatasetOutput
from src.data.pretrain import PretrainDataset
from src.model.model import GNNLLMOutput

if TYPE_CHECKING:
    BaseClass = Trainer
else:
    BaseClass = object  # 运行时不继承Trainer


class DataloaderMixin(BaseClass):

    def eval_collator(self, **kwargs):
        pprint(kwargs)

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:

        dataset = EvaluateDataset(eval_dataset, self.tokenizer, self.cfg)

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": EvaluateDataset.collate_fn,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        eval_dataloader = DataLoader(dataset, **dataloader_params)

        return self.accelerator.prepare(eval_dataloader)

    def train_collator(self, **kwargs):
        pprint(kwargs)

    def get_train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset

        dataset = PretrainDataset(train_dataset, self.tokenizer, self.cfg)

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": PretrainDataset.collate_fn,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "num_workers": 0,
        }

        return self.accelerator.prepare(DataLoader(dataset, **dataloader_params))

    def test_collator(self, **kwargs):
        pprint(kwargs)

    def get_test_dataloader(self, test_dataset: Optional[Dataset] = None) -> DataLoader:
        if not self.cfg.model.only_llm:
            dataset = EvaluateDataset(test_dataset, self.tokenizer, self.cfg)

            dataloader_params = {
                "batch_size": self.args.eval_batch_size,
                "collate_fn": EvaluateDataset.collate_fn,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False
            }
        else:
            dataset = EvaluateLLMMatchEmbDataset(test_dataset, self.tokenizer, self.cfg)

            dataloader_params = {
                "batch_size": self.args.eval_batch_size,
                "collate_fn": EvaluateLLMMatchEmbDataset.collate_fn,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False
            }

        eval_dataloader = DataLoader(dataset, **dataloader_params)

        return self.accelerator.prepare(eval_dataloader)
    
    def get_instruct_dataloader(self, inputs: PretrainDatasetOutput, outputs: GNNLLMOutput) -> DataLoader:
        if not self.cfg.model.only_llm:
            dataset = LPInstrucDataset(inputs.mask_triples, outputs.ent_emb, outputs.rel_emb, self.tokenizer, max_length=self.cfg.task.instruct_len)

            dataloader_params = {
                "batch_size": self.cfg.train.instruct_batch_size,
                "collate_fn": LPInstrucDataset.collate_fn,
                "num_workers": 0,
                "pin_memory": False,
            }
        else:
            dataset = BaseTaskInstrucDataset(inputs.mask_triples, outputs.ent_emb, outputs.rel_emb, self.tokenizer, max_length=self.cfg.task.instruct_len, id_text_maps=inputs._id_text_maps)

            dataloader_params = {
                "batch_size": self.cfg.train.instruct_batch_size,
                "collate_fn": BaseTaskInstrucDataset.collate_fn,
                "num_workers": 0,
                "pin_memory": False,
            }

        # We use the same batch_size as for eval.
        # return self.accelerator.prepare(DataLoader(dataset, **dataloader_params))
        return DataLoader(dataset, **dataloader_params)
