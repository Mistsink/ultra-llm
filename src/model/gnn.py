from dataclasses import asdict
from typing import Optional
import torch
import torch.nn as nn

from config.config import Config, ModelDetailConfig
from src.data.types import ModelInput, PretrainDatasetOutput
from src.ultra.models import EntityNBFNet, RelNBFNet


class GNNModel(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()

        if isinstance(cfg.model.relation_model, ModelDetailConfig):
            rel_cfg = asdict(cfg.model.relation_model)
        else:
            rel_cfg = cfg.model.relation_model
            cfg.model.relation_model = ModelDetailConfig(**rel_cfg)

        if isinstance(cfg.model.entity_model, ModelDetailConfig):
            ent_cfg = asdict(cfg.model.entity_model)
        else:
            ent_cfg = cfg.model.entity_model
            cfg.model.entity_model = ModelDetailConfig(**ent_cfg)

        self.relation_model = RelNBFNet(**rel_cfg)
        self.entity_model = EntityNBFNet(**ent_cfg)

        
    def forward(self, data: ModelInput, boundary: torch.Tensor, layer_idx: int=-1):

        if layer_idx == -1:
            # rel gnn
            # TODO 因为多个batch 中的 r-graph 都是一样的，使用第 0 个生成整个 r-gnn 的emb
            representations = self.relation_model(data.data[0], boundary=boundary[0].unsqueeze(0))
        else:
            # ent gnn
            # batch shape: (bs, 1+num_negs, 3)
            # relations are the same all positive and negative triples, so we can extract only one from the first triple among 1+nug_negs
            # 暂时 bs只为1，data.data 只取 0 位即可
            # TODO data.rel_emb 中需要插入或者替换掉 data.data[0] 中 super_edge_type 的权重
            # data.rel_emb.shape: (1, num_rel, out_dim)
            if layer_idx == 0:
                super_edge_emb = torch.sum(data.rel_emb, dim=1, keepdim=True)    # sum / mean
                if data.rel_emb.shape[1] > data.data[0].super_edge_type[1]:
                    # 直接覆盖
                    data.rel_emb[0, data.data[0].super_edge_type] = super_edge_emb
                elif data.rel_emb.shape[1] <= data.data[0].super_edge_type[1]:
                    # 覆盖和添加
                    num_gap = data.data[0].super_edge_type[1] + 1 - data.rel_emb.shape[1]
                    rel_emb = torch.cat([data.rel_emb, torch.zeros(1, num_gap, data.rel_emb.shape[2], device=data.rel_emb.device, dtype=data.rel_emb.dtype)], dim=1)
                    rel_emb[0, data.data[0].super_edge_type] = super_edge_emb
                    data.rel_emb = rel_emb
                
                
            representations = self.entity_model(data.data[0], data.rel_emb, data.mask_triples, boundary, layer_idx)

        return representations