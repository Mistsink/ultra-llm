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
            representations = self.relation_model(data.data, boundary=boundary)
        else:
            # ent gnn
            # batch shape: (bs, 1+num_negs, 3)
            # relations are the same all positive and negative triples, so we can extract only one from the first triple among 1+nug_negs

            representations = self.entity_model(data.data, data.rel_emb, data.mask_triples, boundary, layer_idx)

        return representations