from pprint import pprint
from typing import TYPE_CHECKING, Optional
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer

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
        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": self.test_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        # We use the same batch_size as for eval.
        return self.accelerator.prepare(DataLoader(test_dataset, **dataloader_params))
    
    def get_instruct_dataloader(self, inputs: PretrainDatasetOutput, outputs: GNNLLMOutput) -> DataLoader:
        dataset = LPInstrucDataset(inputs.mask_triples, outputs.ent_emb, outputs.rel_emb, self.tokenizer, max_length=self.cfg.task.instruct_len)

        dataloader_params = {
            "batch_size": self.cfg.train.instruct_batch_size,
            "collate_fn": LPInstrucDataset.collate_fn,
            "num_workers": 0,
            "pin_memory": False,
        }

        # We use the same batch_size as for eval.
        # return self.accelerator.prepare(DataLoader(dataset, **dataloader_params))
        return DataLoader(dataset, **dataloader_params)
