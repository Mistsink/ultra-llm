from dataclasses import dataclass
from typing import Optional
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

    def __len__(self):
        return self.data.num_graphs


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
        return ModelInput(data.data_rel, data.mask_triples, data.rel_prompt, data.rel_begin_idx, data.rel_end_idx, data.rel_ranges, is_ent=False)
    
    @staticmethod
    def ent_from_pretrain_output(data: PretrainDatasetOutput, rel_emb: Optional[torch.Tensor] = None):
        return ModelInput(data.data, data.mask_triples, data.ent_prompt, data.ent_begin_idx, data.ent_end_idx, data.ent_ranges, is_ent=True, rel_emb=rel_emb)
        
