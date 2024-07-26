from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertGenerationEncoder, BertLayer, BertConfig
from transformers.models.bert.modeling_bert import BertEncoder
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa

from config.config import Config


class Encoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.cfg = config
        bert_cfg = BertConfig(
            hidden_size=config.model.trans_hidden_dim,
            num_hidden_layers=config.model.trans_layers,
            num_attention_heads=config.model.trans_heads,
            hidden_dropout_prob=config.model.trans_dropout,
            attention_probs_dropout_prob=config.model.trans_dropout,
            max_position_embeddings=config.model.trans_max_length,
            layer_norm_eps=config.model.trans_eps,
        )
        self.encoder = BertEncoder(bert_cfg)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        for i, layer in enumerate(self.encoder.layer):
            while (
                torch.isnan(
                    self.encoder.layer[i].attention.output.LayerNorm.weight
                ).any()
                or torch.isnan(
                    self.encoder.layer[i].attention.output.LayerNorm.bias
                ).any()
            ):
                self.encoder.layer[i].attention.output.LayerNorm = nn.LayerNorm(
                    self.cfg.model.trans_hidden_dim, eps=self.cfg.model.trans_eps
                ).to(
                    device=hidden_states.device,
                    dtype=self.encoder.layer[i].attention.output.LayerNorm.weight.dtype,
                )
            while (
                torch.isnan(self.encoder.layer[i].output.LayerNorm.weight).any()
                or torch.isnan(self.encoder.layer[i].output.LayerNorm.bias).any()
            ):
                self.encoder.layer[i].output.LayerNorm = nn.LayerNorm(
                    self.cfg.model.trans_hidden_dim, eps=self.cfg.model.trans_eps
                ).to(
                    device=hidden_states.device,
                    dtype=self.encoder.layer[i].output.LayerNorm.weight.dtype,
                )
        return self.encoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )


class TransEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.cfg = config

        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)

    def forward(
        self, pre_embs: torch.Tensor, options_embs: torch.Tensor
    ) -> Tuple[torch.Tensor, BaseModelOutputWithPastAndCrossAttentions]:
        """
        pre_embs: [1, 2, embes, dim]
        options_embs: [1, num_options, embes, dim]
        """
        batch_size = pre_embs.size(0)
        assert batch_size == 1
        device = pre_embs.device

        # dim: 1, seq_len, dim
        input_emb, options_position_ids = self.embeddings(
            pre_embs=pre_embs,
            options_embs=options_embs,
        )
        seq_length = input_emb.size(1)

        attention_mask = torch.ones((batch_size, seq_length), device=device)

        # Expand the attention mask
        # Expand the attention mask for SDPA.
        # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
        extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
            attention_mask, input_emb.dtype, tgt_len=seq_length
        )

        encoder_out: BaseModelOutputWithPastAndCrossAttentions = self.encoder(
            hidden_states=input_emb,
            attention_mask=extended_attention_mask,
            use_cache=False,
        )

        logits = self.compute_logits(
            encoder_out.last_hidden_state, options_position_ids
        )
        return logits, encoder_out

    def compute_logits(
        self, seq_emb: torch.Tensor, options_position_ids: List[List[int]]
    ) -> torch.Tensor:
        """
        seq_emb: [1, seq_len, dim]
        """
        seq_emb = seq_emb.squeeze(0)
        task = seq_emb[0]
        logits = []
        dim_sqrt = torch.sqrt(
            torch.tensor(self.cfg.model.trans_hidden_dim, device=seq_emb.device)
        )
        for option_pos in options_position_ids:
            option_emb = seq_emb[option_pos]  # [num_embs, dim]
            option_emb = torch.cat(
                [option_emb, torch.mean(option_emb, dim=0).unsqueeze(0)], dim=0
            )  # [num_embs+1, dim]
            # option_emb = torch.mean(option_emb, dim=0)  # dim: dim

            # _logit = torch.matmul(task, option_emb) / dim_sqrt

            _logits = torch.matmul(task, option_emb.T) / dim_sqrt
            __logits = F.gumbel_softmax(_logits, hard=True)
            _logit = torch.matmul(_logits, __logits.T)
            logits.append(_logit)  # dim: []

        logits = torch.stack(logits, dim=0)  # [num_options]
        return logits.unsqueeze(0)


class Embeddings(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.cfg = config

        tokens = ["TASK", "ENTITY", "RELATION", "[SEP]"]
        self.token2id = {token: torch.tensor(i) for i, token in enumerate(tokens)}
        self.special_token_embeddings = nn.Embedding(
            len(tokens), self.cfg.model.trans_hidden_dim
        )

        self.position_embeddings = nn.Embedding(
            self.cfg.model.trans_max_length, self.cfg.model.trans_hidden_dim
        )
        self.LayerNorm = nn.LayerNorm(
            self.cfg.model.trans_hidden_dim, eps=self.cfg.model.trans_eps
        )
        self.dropout = nn.Dropout(self.cfg.model.trans_dropout)

    def SEPEmb(self):
        return self.special_token_embeddings(self.token2id["[SEP]"])

    def TaskEmb(self):
        return self.special_token_embeddings(self.token2id["TASK"])

    def EntityEmb(self):
        return self.special_token_embeddings(self.token2id["ENTITY"])

    def RelationEmb(self):
        return self.special_token_embeddings(self.token2id["RELATION"])

    def forward(
        self, pre_embs: torch.Tensor, options_embs: torch.Tensor
    ) -> torch.Tensor:
        """
        pre_embs: [1, 2, embes, dim]
        options_embs: [1, num_options, embes, dim]
        """
        # [T] [E] pre_embs[:-1] [R] pre_embs[-1] [SEP] [E] options_emb_1 [E] options_emb_2 ...
        sep_emb = self.SEPEmb().unsqueeze(0)
        task_emb = self.TaskEmb().unsqueeze(0)
        entity_emb = self.EntityEmb().unsqueeze(0)
        relation_emb = self.RelationEmb().unsqueeze(0)

        first_part = torch.cat(
            [task_emb, entity_emb, pre_embs[0, 0], relation_emb], dim=0
        )
        last_part = pre_embs[0, 1]

        option_part, options_position_ids = [], []
        start_idx = len(first_part) + len(last_part) + 1
        for option_embs in options_embs[0]:
            _option_part = [entity_emb]
            _option_part.append(option_embs)

            options_position_ids.append(
                list(range(start_idx, start_idx + len(option_embs) + 1))
            )
            start_idx += len(option_embs) + 1

            option_part.extend(_option_part)

        full_emb = torch.cat(
            [first_part, last_part, sep_emb, *option_part], dim=0
        ).unsqueeze(
            0
        )  # dim: 1, seq_len, dim

        position_ids = torch.arange(full_emb.size(1), device=full_emb.device).unsqueeze(
            0
        )
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = full_emb + position_embeddings

        while (
            torch.isnan(self.LayerNorm.weight).any()
            or torch.isnan(self.LayerNorm.bias).any()
        ):
            self.LayerNorm = nn.LayerNorm(
                self.cfg.model.trans_hidden_dim, eps=self.cfg.model.trans_eps
            ).to(device=embeddings.device, dtype=self.LayerNorm.weight.dtype)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, options_position_ids
