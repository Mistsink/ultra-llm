from dataclasses import dataclass
from typing import List, Optional, Union
import torch
import torch.nn as nn
from transformers import GemmaForCausalLM, GemmaTokenizer, GemmaConfig, AutoModelForCausalLM, LlamaForCausalLM, LlamaConfig
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
)

from config.config import Config
from src.model.layer import MLP
from src.data.types import PretrainDatasetOutput, ModelInput
from src.ultra.models import RelNBFNet, EntityNBFNet
from src.model.gemma_gnn import GNNLLMModel, GNNLLMModelOutput
from src.model.type import GNNLLMConfig
from src.model.transformer import TransEncoder


@dataclass
class GNNLLMOutput(CausalLMOutputWithPast):
    ent_emb: Optional[torch.Tensor] = None
    rel_emb: Optional[torch.Tensor] = None


class GNNLLM(LlamaForCausalLM):
    config_class = GNNLLMConfig
    def __init__(self, config: LlamaConfig, cfg: Config):
        super().__init__(config)

        self.dummy_layer = nn.Linear(3, 3, bias=False)

        self.model = GNNLLMModel(config, cfg=cfg)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.proj_rel_layer = MLP(
            cfg.model.relation_model.hidden_dims[-1],
            config.hidden_size,
            config.hidden_size,
            2,
        )

        # Initialize weights and apply final processing
        self.post_init()

        self.transformer_decoder = TransEncoder(cfg)

    def init_graph_tokenizer(self, tokenizer: GemmaTokenizer, num_new_tokens: int):
        self.model.init_graph_tokenizer(tokenizer, num_new_tokens)
        # self.resize_token_embeddings(len(tokenizer))

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        data: Optional[PretrainDatasetOutput] = None,
        embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[CausalLMOutputWithPast, GNNLLMOutput]:
        output_attentions, output_hidden_states, return_dict = self._forward_setup(output_attentions, output_hidden_states, return_dict)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        if data is not None:
            # As encoder

            # First: encode rel emb
            # rel_input = ModelInput.rel_from_pretrain_output(data)
            # rel_out: GNNLLMModelOutput = self.model(
            #     attention_mask=attention_mask,
            #     position_ids=position_ids,
            #     past_key_values=past_key_values,
            #     inputs_embeds=inputs_embeds,
            #     use_cache=use_cache,
            #     output_attentions=output_attentions,
            #     output_hidden_states=output_hidden_states,
            #     return_dict=return_dict,
            #     cache_position=cache_position,
            #     # Valid vars
            #     input_ids=rel_input.prompt,
            #     model_input=rel_input,
            # )

            # Second: encode ent emb
            # ent_input = ModelInput.ent_from_pretrain_output(
            #     data, rel_emb=rel_out.rel_emb
            # )
            ent_input = ModelInput.ent_from_pretrain_output(
                data, rel_emb=None
            )
            ent_out: GNNLLMModelOutput = self.model(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                # Valid vars
                input_ids=ent_input.prompt,
                model_input=ent_input,
            )

            # project rel_emb from gnn_dim into hidden_dim
            # rel_out.rel_emb = self.proj_rel_layer(rel_out.rel_emb)

            # return GNNLLMOutput(ent_emb=ent_out.ent_emb, rel_emb=rel_out.rel_emb)
            return GNNLLMOutput(ent_emb=ent_out.ent_emb, rel_emb=None)

        else:
            assert embeds is not None, "embeds must be provided when data is None, as to model is a decoder"
            # # LLM As decoder
            # outputs: BaseModelOutputWithPast = self.model(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     position_ids=position_ids,
            #     past_key_values=past_key_values,
            #     inputs_embeds=inputs_embeds,
            #     use_cache=use_cache,
            #     output_attentions=output_attentions,
            #     output_hidden_states=output_hidden_states,
            #     return_dict=return_dict,
            #     cache_position=cache_position,
            #     embeds=embeds
            # )

            ###########################################
            ####        transformer decoder        ####
            ###########################################
            # pre_embs 1, embes, dim
            # options_embs 1, num_options, embes, dim
            pre_embs = embeds[:2].unsqueeze(0)
            options_embs = embeds[2:].unsqueeze(0).unsqueeze(2)
            logits, hidden_out = self.transformer_decoder(pre_embs=pre_embs, options_embs=options_embs)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

            # TODO FIXME 这里是为了让返回的 logits 数据变少，否则在 eval 时外部会一直保留积累每个 batch 的 logits，送给 compute_metric fn 中使用
            logits = logits.argmax(-1).unsqueeze(0)  # [labels != -100]
        else:
            logits = logits.float()
            loss = None

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            # past_key_values=outputs.past_key_values,
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
        )

    def _forward_setup(self, output_attentions, output_hidden_states, return_dict):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        return output_attentions, output_hidden_states, return_dict


    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs
        )

        # TODO need :param embeds
        model_inputs['embeds'] = kwargs['embeds']
        return model_inputs

class TestModel(nn.Module):
    def __inin__(self):
        pass

    def forward(self, data, batch):
        return torch.ones(batch.shape[0], 1)


if __name__ == "__main__":
    m = GNNLLM(None, None)
    m.generate()