from pprint import pprint
from typing import TYPE_CHECKING, Optional
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer

from src.data.pretrain import PretrainDataset

if TYPE_CHECKING:
    BaseClass = Trainer
else:
    BaseClass = object  # 运行时不继承Trainer


class DataloaderMixin(BaseClass):

    def eval_collator(self, **kwargs):
        pprint(kwargs)

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": self.eval_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)

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
