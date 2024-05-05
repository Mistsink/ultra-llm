from dataclasses import dataclass
from typing import List, Optional, Union
import torch
import torch.nn as nn
from transformers import GemmaForCausalLM, GemmaTokenizer, GemmaConfig
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
)

from config.config import Config
from src.data.types import PretrainDatasetOutput, ModelInput
from src.ultra.models import RelNBFNet, EntityNBFNet
from src.model.gemma_gnn import GNNLLMModel, GNNLLMModelOutput


@dataclass
class GNNLLMOutput(CausalLMOutputWithPast):
    ent_emb: Optional[torch.Tensor] = None
    rel_emb: Optional[torch.Tensor] = None


class GNNLLM(GemmaForCausalLM):
    def __init__(self, config: GemmaConfig, cfg: Config):
        super().__init__(config)
        self.model = GNNLLMModel(config, cfg=cfg)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def init_graph_tokenizer(self, tokenizer: GemmaTokenizer, num_new_tokens: int):
        self.model.init_graph_tokenizer(tokenizer, num_new_tokens)
        self.resize_token_embeddings(len(tokenizer))

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
        output_attentions, output_hidden_states, return_dict = self._forward_setup()

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        if data is not None:
            # As encoder

            # First: encode rel emb
            rel_input = ModelInput.rel_from_pretrain_output(data)
            rel_out: GNNLLMModelOutput = self.model(
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
                input_ids=rel_input.prompt,
                model_input=rel_input,
            )

            # Second: encode ent emb
            ent_input = ModelInput.ent_from_pretrain_output(
                data, rel_emb=rel_out.rel_emb
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

            return GNNLLMOutput(ent_emb=ent_out.ent_emb, rel_emb=rel_out.rel_emb)

        else:
            assert embeds is not None, "embeds must be provided when data is None, as to model is a decoder"
            # As decoder
            outputs: BaseModelOutputWithPast = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                embeds=embeds
            )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        loss = None

        # TODO Instruction Tuning
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _forward_setup(self):
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


class TestModel(nn.Module):
    def __inin__(self):
        pass

    def forward(self, data, batch):
        return torch.ones(batch.shape[0], 1)
