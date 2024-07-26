from dataclasses import dataclass
from typing import Optional, Union
import torch
from torch_geometric.data import Data, Batch

from src.data.graph_text_store import TextData


class CustomData(Data):
    target_edge_index: torch.Tensor
    target_edge_type: torch.Tensor
    entity_vocab: Optional[dict[str, int]]
    relation_vocab: Optional[dict[str, int]]
    text_data: TextData
    relation_graph: Optional[Data]
    num_relations: int


class CustomSubData(CustomData):
    n_id: torch.Tensor
    e_id: torch.Tensor
    edge_label_index: torch.Tensor


class CustomSubDataWithSuperNode(CustomSubData):
    super_node_id: int
    super_edge_type: torch.Tensor
    begin_super_edge_index: int


@dataclass
class PretrainDatasetItemOutput:
    mask_triples: torch.Tensor
    data: CustomSubDataWithSuperNode
    ent_prompt: torch.Tensor
    rel_prompt: torch.Tensor

    # rel idx in rel_prompt
    rel_begin_idx: torch.Tensor
    rel_end_idx: torch.Tensor
    rel_ranges: list[torch.Tensor]

    # ent idx in ent_prompt
    ent_begin_idx: torch.Tensor
    ent_end_idx: torch.Tensor
    ent_ranges: list[torch.Tensor]

    _labels: Optional[torch.Tensor]=None
    _id_text_map: Optional[dict[int, str]]=None

    # for debug
    ht_id: Optional[list[int]]=None
    neg_t_ids: Optional[list[int]]=None


@dataclass
class PretrainDatasetOutput:
    mask_triples: torch.Tensor
    data: Batch
    data_rel: Batch
    ent_prompt: torch.Tensor
    rel_prompt: torch.Tensor

    # rel idx in rel_prompt
    rel_begin_idx: torch.Tensor
    rel_end_idx: torch.Tensor
    rel_ranges: list[torch.Tensor]

    # ent idx in ent_prompt
    ent_begin_idx: torch.Tensor
    ent_end_idx: torch.Tensor
    ent_ranges: list[torch.Tensor]

    _labels: Optional[torch.Tensor]=None
    _id_text_maps: Optional[list[dict[int, str]]]=None

    def __len__(self):
        return self.data.num_graphs

    def to(self, device):
        self.data = self.data.to(device)
        self.data_rel = self.data_rel.to(device)
        self.ent_prompt = self.ent_prompt.to(device)
        self.rel_prompt = self.rel_prompt.to(device)

        self.rel_begin_idx = self.rel_begin_idx.to(device)
        self.rel_end_idx = self.rel_end_idx.to(device)
        self.rel_ranges = [i.to(device) for i in self.rel_ranges]

        self.ent_begin_idx = self.ent_begin_idx.to(device)
        self.ent_end_idx = self.ent_end_idx.to(device)
        self.ent_ranges = [i.to(device) for i in self.ent_ranges]

        if self._labels is not None:
            self._labels = self._labels.to(device)

        return self
    
    def __contains__(self, item):
        if item == 'input_ids':
            return True
        else:
            return False

    def __getitem__(self, key):
        if key == 'input_ids':
            return torch.stack([self.ent_prompt, self.rel_prompt])
        
        return None
        
    def get(self, key, default_val: Optional[any]=None):
        if key == 'labels':
            return default_val if self._labels is None else self._labels
        elif key == 'return_loss':
            return True
        return default_val


@dataclass
class ModelInput:
    data: Batch
    mask_triples: torch.Tensor
    prompt: torch.Tensor
    g_begin_idx: torch.Tensor
    g_end_idx: torch.Tensor
    ranges: list[torch.Tensor]
    is_ent: bool
    rel_emb: Optional[torch.Tensor] = None

    def __len__(self):
        return self.data.num_graphs

    @staticmethod
    def rel_from_pretrain_output(data: PretrainDatasetOutput):
        return ModelInput(
            data.data_rel,
            data.mask_triples,
            data.rel_prompt,
            data.rel_begin_idx,
            data.rel_end_idx,
            data.rel_ranges,
            is_ent=False,
        )

    @staticmethod
    def ent_from_pretrain_output(
        data: PretrainDatasetOutput, rel_emb: Optional[torch.Tensor] = None
    ):
        return ModelInput(
            data.data,
            data.mask_triples,
            data.ent_prompt,
            data.ent_begin_idx,
            data.ent_end_idx,
            data.ent_ranges,
            is_ent=True,
            rel_emb=rel_emb,
        )


@dataclass
class InstrucInput:
    triple: Union[torch.Tensor, list[int], tuple[int, int, int]]
    input_ids: torch.Tensor
    label_ids: torch.Tensor
    embs: list[torch.Tensor]  # [ent_embs..., rel_emb]
    g_emb: Optional[torch.Tensor] = None

    def to(self, device):
        if self.input_ids is not None:
            self.input_ids = self.input_ids.to(device)
        self.label_ids = self.label_ids.to(device)
        self.embs = self.embs.to(device)
        if self.g_emb is not None:
            self.g_emb = self.g_emb.to(device)

        return self


def find_subsequence_in_list(
    lst: list[str], subseq: list[str], occurrence=1, start_index=0
):
    subseq_length = len(subseq)  # 子序列的长度
    if start_index < 0 or start_index >= len(lst):
        return -1  # 如果起始索引无效，直接返回 -1

    max_index = len(lst) - subseq_length + 1  # 可以检查子序列的最大起始索引
    count = 0  # 用于计数找到的子序列次数

    # 从指定的起始索引开始遍历
    for i in range(start_index, max_index):
        # 使用切片检查从当前索引开始的子列表是否与目标子序列匹配
        if lst[i : i + subseq_length] == subseq:
            count += 1
            if count == occurrence:
                return i  # 找到第 `occurrence` 次出现，返回子序列的起始索引

    return -1  # 如果列表中没有找到指定次数的子序列，返回 -1
