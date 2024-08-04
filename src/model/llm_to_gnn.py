from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    GemmaForCausalLM,
    LlamaForCausalLM,
    AutoModel,
    LlamaModel,
    LlamaConfig,
    GemmaModel,
    GemmaConfig,
    GemmaTokenizer,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.gemma.modeling_gemma import logger
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.cache_utils import StaticCache, DynamicCache, Cache

from config.config import Config
from src.data.special_tokens import SpecialToken
from src.data.types import CustomSubDataWithSuperNode, ModelInput
from src.model.layer import MLP
from src.model.gnn import GNNModel
from src.model.type import GNNLLMConfig


@dataclass
class GNNLLMModelOutput(BaseModelOutputWithPast):
    ent_emb: Optional[torch.Tensor] = None
    rel_emb: Optional[torch.Tensor] = None


class LLM2GNNModel(LlamaModel):
    config_class = GNNLLMConfig

    def __init__(self, config: LlamaConfig, cfg: Config):
        super().__init__(config)
        self.cfg = cfg

        self.encoder_layers_num = len(self.layers)
        self.interact_layers_num = len(cfg.model.entity_model.hidden_dims)

        self.graph_model = self.__init_graph_models(cfg)

        self.exchange_info_layer = self.__init_exchange_info_layer(config, cfg)
        self.llm_to_rel_layer = MLP(
            config.hidden_size * 2,
            config.hidden_size,
            cfg.model.relation_model.input_dim,
            2,
        )
        self.llm_to_ent_layer = MLP(
            config.hidden_size * 2,
            config.hidden_size,
            cfg.model.entity_model.input_dim,
            2,
        )

        if self.cfg.model.only_llm:
            self.fuse_llm_ent_layer = MLP(
                config.hidden_size * 2,
                cfg.model.trans_hidden_dim,
                cfg.model.trans_hidden_dim,
                2,
            )
            self.fuse_llm_rel_layer = MLP(
                config.hidden_size * 2,
                cfg.model.trans_hidden_dim,
                cfg.model.trans_hidden_dim,
                2,
            )
        else:
            self.fuse_llm_ent_layer = MLP(
                config.hidden_size * 2 + cfg.model.entity_model.hidden_dims[-1],
                cfg.model.trans_hidden_dim,
                cfg.model.trans_hidden_dim,
                # config.hidden_size,
                2,
            )
            self.fuse_llm_rel_layer = MLP(
                config.hidden_size * 2 + cfg.model.relation_model.hidden_dims[-1],
                cfg.model.trans_hidden_dim,
                cfg.model.trans_hidden_dim,
                # config.hidden_size,
                2,
            )

        # self.ent_norm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ent_norm = LlamaRMSNorm(cfg.model.trans_hidden_dim, config.rms_norm_eps)

        # 用于将融合后的 ent_emb 映射到原始 token 的 emb by gumble softmax
        self.ori_vocab_size = config.vocab_size
        # self.fused_ent_token_to_ori_token = nn.Linear(
        #     config.hidden_size, config.vocab_size
        # )

    def __init_graph_models(self, config: Config) -> GNNModel:
        return GNNModel(config)

    def __init_exchange_info_layer(
        self, config: GemmaConfig, gnn_config: Config
    ) -> nn.Module:
        return MLP(
            config.hidden_size + gnn_config.model.entity_model.hidden_dims[-1],
            config.hidden_size,
            config.hidden_size + gnn_config.model.entity_model.hidden_dims[-1],
            1,
        )

    def init_graph_tokenizer(self, tokenizer: GemmaTokenizer, num_new_tokens: int = 4):
        """
        为了 __is_special_token 做准备
        """
        self.tokenizer = tokenizer
        self.special_id_map = SpecialToken.get_token_ids(tokenizer)
        self.special_ids = list(self.special_id_map.values())
        self.resize_token_embeddings(len(tokenizer))

        # custom Embedding for special tokens
        self.special_input_emb = nn.Embedding(
            num_new_tokens, self.embed_tokens.weight.size(1)
        ).to(self.device)
        self.id_to_special_index = {
            token_id: i for i, token_id in enumerate(self.special_ids)
        }

    def __is_special_token(self, token_id: int) -> bool:
        return token_id in self.special_ids

    @staticmethod
    def get_super_node_idxs(g: CustomSubDataWithSuperNode) -> torch.Tensor:
        """
        FIXME: 可以在NodeGraphIdxMapBatched中实现
        :return: super node idxs [int]
        """
        num_nodes_per_graph = g.batch_num_nodes()

        super_node_indices = []
        node_count = 0
        for num_nodes in num_nodes_per_graph:
            # 超级节点是每个图的最后一个节点
            super_node_idx = node_count + num_nodes - 1
            super_node_indices.append(super_node_idx)
            node_count += num_nodes

        return torch.stack(super_node_indices)

    def _setup_and_validate_inputs(
        self,
        input_ids,
        inputs_embeds,
        attention_mask,
        use_cache,
        past_key_values,
        cache_position,
        position_ids,
        output_attentions,
        model_input: Optional[ModelInput] = None,
        embeds: Optional[torch.FloatTensor] = None,
    ):
        output_attentions = (
            self.config.output_attentions
            if output_attentions is None
            else output_attentions
        )
        output_hidden_states = self.config.output_hidden_states
        use_cache = self.config.use_cache
        return_dict = self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify either input_ids or inputs_embeds, not both."
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # generate inputs_embeds
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            # >>> Replace special token embeddings <<<
            if model_input is not None:
                inputs_embeds = self.__replace_common_special_token_emb(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    model_input=model_input,
                )

            # >>> Replace info token embeddings when as decoder <<<
            if embeds is not None:
                inputs_embeds = self.__replace_info_token_emb(
                    input_ids=input_ids, inputs_embeds=inputs_embeds, embeds=embeds
                )

        past_seen_tokens = 0
        if use_cache:
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
        )

        normalizer = torch.tensor(
            self.config.hidden_size**0.5, dtype=inputs_embeds.dtype
        )
        inputs_embeds = inputs_embeds * normalizer

        return (
            inputs_embeds,
            causal_mask,
            position_ids,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

    def __replace_common_special_token_emb(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        model_input: Optional[ModelInput] = None,
    ) -> torch.Tensor:
        """
        原本的词表是不可学习的，这里将新增的特殊 token 的 emb 替换为可学习的 emb
        :return: replaced inputs_embeds
        """
        token = self.tokenizer.tokenize(SpecialToken.G_BEGIN.value)[0]
        _id = self.tokenizer.convert_tokens_to_ids(token)
        __idx = torch.tensor(self.id_to_special_index[_id]).to(
            self.special_input_emb.weight.device
        )
        inputs_embeds[:, model_input.g_begin_idx] = self.special_input_emb(__idx)

        token = self.tokenizer.tokenize(SpecialToken.G_END.value)[0]
        _id = self.tokenizer.convert_tokens_to_ids(token)
        __idx = torch.tensor(self.id_to_special_index[_id]).to(
            self.special_input_emb.weight.device
        )
        inputs_embeds[:, model_input.g_end_idx] = self.special_input_emb(__idx)
        return inputs_embeds

    def __replace_info_token_emb(
        self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor, embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        将 info token 的 emb 替换为可学习的 embeds 中的 emb
        :return: replaced inputs_embeds
        """

        # TODO 是否有专门的 Linear 分别处理 ent、rel
        # embeds 转换成离散的 emb_tokens 的加权组合
        # 使用 gumble softmax 来实现
        # 1) 使用 Linear
        # logits = self.fused_ent_token_to_ori_token(embeds)
        # 2) 计算点积相似度
        logits = torch.matmul(
            embeds, self.embed_tokens.weight[: self.ori_vocab_size].t()
        )

        # logits: (1, num_tokens, vocab_size)
        # 从 logits 中采样出 token
        # hard = True 时，采样出的 token 是 one-hot 的, hard = False 时，采样出的 token 是 softmax 的
        logits = F.gumbel_softmax(logits, hard=True)
        embeds = torch.matmul(logits, self.embed_tokens.weight[: self.ori_vocab_size])

        for i in range(input_ids.size(0)):
            _input_ids = input_ids[i]
            _inputs_embed = inputs_embeds[i]
            _cnt = 0
            for j, _id in enumerate(_input_ids):  # TODO FIXME 太耗时了
                _id = _id.cpu().item()
                if _id not in self.id_to_special_index:
                    continue
                inputs_embeds[i][j] = embeds[i][_cnt]
                _cnt += 1
                if _cnt == len(embeds[i]):
                    break

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        model_input: Optional[ModelInput] = None,
        embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[BaseModelOutputWithPast, GNNLLMModelOutput]:
        (
            inputs_embeds,
            causal_mask,
            position_ids,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        ) = self._setup_and_validate_inputs(
            input_ids,
            inputs_embeds,
            attention_mask,
            use_cache,
            past_key_values,
            cache_position,
            position_ids,
            output_attentions,
            model_input=model_input,
            embeds=embeds,
        )
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            # TODO E-GNN & LLM interaction
            if (
                model_input is not None
                and i >= self.encoder_layers_num - self.interact_layers_num
            ):
                g_emb, hidden_states = self.interact_gnn_layer(
                    model_input, hidden_states, i, model_input.is_ent
                )

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # R-GNN & LLM generate rel emb
        if model_input is not None:
            if self.cfg.model.only_llm:
                g_emb = None
            fused_emb = self.fuse_llm_gnn(model_input, hidden_states, g_emb)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if isinstance(next_decoder_cache, Cache)
                else next_decoder_cache
            )

        if model_input is not None:
            return GNNLLMModelOutput(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
                ent_emb=fused_emb if model_input.is_ent else None,
                rel_emb=fused_emb if not model_input.is_ent else None,
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def fuse_llm_gnn(
        self,
        model_input: ModelInput,
        hidden_states: torch.Tensor,
        g_state: torch.Tensor,
    ) -> torch.Tensor:
        bs = len(model_input.ranges)
        assert bs == 1, "暂时只支持 batch_size = 1, 否则显存会爆"
        boundary = self.get_boundary(
            model_input, hidden_states, is_entity=model_input.is_ent
        )

        if self.cfg.model.only_llm:
            # 没有 GNN 模块
            if model_input.is_ent:
                feat = self.fuse_llm_ent_layer(boundary)
            else:
                feat = self.fuse_llm_rel_layer(boundary)
            return feat

        feat = torch.cat([boundary, g_state], dim=2)  # TODO FIXME dim ?
        if model_input.is_ent:
            feat = self.fuse_llm_ent_layer(feat)
        else:
            feat = self.fuse_llm_rel_layer(feat)
        return feat

    def interact_gnn_layer(
        self,
        model_input: ModelInput,
        hidden_states: torch.Tensor,
        i: int,
        is_entity: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        llm 信息注入到 gnn 中, 返回更新后的 gnn_state, hidden_states
        """
        idx = i - (self.encoder_layers_num - self.interact_layers_num)

        # 将 LLM 输出转为 GNN中的 boundary
        boundary = None
        # 每一层的 boundary 都使用更新后的 LLM_hidden_feat 来构建
        if idx >= 0:
            boundary = self.get_boundary(model_input, hidden_states, is_entity)
            if is_entity:
                boundary = self.llm_to_ent_layer(boundary)
            else:
                boundary = self.llm_to_rel_layer(boundary)

        # GNN
        if is_entity:
            ent_emb = self.interact_ent_gnn(model_input, idx, boundary)
            return ent_emb, hidden_states
        else:
            rel_emb = self.interact_rel_gnn(model_input, idx, boundary)
            return rel_emb, hidden_states

    def get_boundary(self, model_input, hidden_states, is_entity):
        bs = len(model_input.ranges)
        assert bs == 1, "暂时只支持 batch_size = 1, 否则显存会爆"
        while len(model_input.mask_triples.shape) > 3:
            model_input.mask_triples = model_input.mask_triples.squeeze(0)

        num_nodes = model_input.data[0].num_nodes
        if not is_entity:
            # r-gnn 取整个 rel-graph，num_nodes // 2 个节点均有特征
            num_nodes //= 2
        boundary = torch.zeros(
            bs,
            num_nodes,
            hidden_states.shape[2] * 2,
            device=self.device,
            dtype=hidden_states.dtype,
        )
        if is_entity:
            for i, ranges in enumerate(model_input.ranges):
                for _i, j in enumerate(model_input.data[i].n_id):
                    j = _i  # 暂时这样写是因为 entity id 改成子图中的新id，而不是原图中的 id
                    # assert ranges[j][0][0] != 0, "确保每个 entity 都在 LLM 的prompt中出现过"
                    if ranges[j][0][0] == -1:
                        # 该 entity 未在 LLM 的prompt中出现过
                        pass
                    else:
                        boundary[i, _i] = hidden_states[i, ranges[j][0]].view(-1)

                    # 单独处理 super_node
                boundary[i, model_input.data[i].super_node_id] = hidden_states[
                    i, [model_input.g_begin_idx, model_input.g_end_idx]
                ].view(-1)
        else:
            # for r-graph
            for i, ranges in enumerate(model_input.ranges):
                for j in range(model_input.data[i].num_nodes // 2):
                    if ranges[j][0][0] == -1:
                        # 该 relation 未在 LLM 的prompt中出现过
                        pass
                    else:
                        boundary[i, j] = hidden_states[i, ranges[j][0]].view(-1)
                # GNN 中有 real_nodes * 2 * bs 个节点
                # 有逆向边，故 * 2
                # 有 Batch, 故 * bs
                # TODO FIXME 这里简单将 原始边的emb作为 逆向边 的特征
            boundary = boundary.repeat(1, 2, 1)
        return boundary

    def interact_ent_gnn(
        self, model_input: ModelInput, idx: int, boundary: torch.Tensor
    ) -> torch.Tensor:
        """
        :return: ent_emb
        """
        ent_emb = self.graph_model(model_input, boundary=boundary, layer_idx=idx)

        # 仅由 LLM -> boundary => GNN
        return ent_emb

    def interact_rel_gnn(
        self, model_input: ModelInput, idx: int, boundary: torch.Tensor
    ) -> torch.Tensor:
        """
        :return: rel_emb
        """
        rel_emb = self.graph_model(model_input, boundary=boundary, layer_idx=idx)

        return rel_emb

    def emb_from_mean_tokens(
        self, hidden_states: torch.Tensor, model_input: ModelInput
    ):
        """
        ####################################################
        # 测试方案：直接将 text 的 token 拼接池化作为 ent_emb   #
        ####################################################
        """
        bs = len(model_input.ranges)
        assert bs == 1, "暂时只支持 batch_size = 1, 否则显存会爆"
        boundary = torch.zeros(
            bs,
            model_input.data[0].num_nodes,
            hidden_states.shape[2],
            device=self.device,
            dtype=hidden_states.dtype,
        )
        for i, ranges in enumerate(model_input.ranges):
            for _i, j in enumerate(model_input.data[i].n_id):
                j = _i  # 暂时这样写是因为 entity id 改成子图中的新id，而不是原图中的 id
                if ranges[j][0][0] == -1:
                    # 该 entity 未在 LLM 的prompt中出现过
                    pass
                else:
                    # boundary[i, _i] = hidden_states[i, ranges[j][0]].view(-1)
                    l_idx, r_idx = ranges[j][0]  # [l, r]
                    boundary[i, _i] = hidden_states[i, l_idx : r_idx + 1].mean(dim=0)

            # 单独处理 super_node
            boundary[i, model_input.data[i].super_node_id] = hidden_states[
                i, model_input.g_begin_idx : model_input.g_end_idx + 1
            ].mean(dim=0)

        return boundary
