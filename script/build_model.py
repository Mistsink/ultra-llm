from typing import List
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    GemmaTokenizer,
    LlamaTokenizer,
    AutoModelForCausalLM,
    GemmaForCausalLM,
    LlamaForCausalLM,
    BitsAndBytesConfig,
)
import bitsandbytes as bnb

from src.model.model import GNNLLM, TestModel
from src.data.special_tokens import SpecialToken
from src.ultra.models import Ultra
from config.config import Config

from peft import (
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftConfig
)


def find_all_linear_names(model: GemmaForCausalLM, cfg: Config):
    """
    找出所有与 GNN 交互的层 的线性层
    """

    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    layers_num = len(model.model.layers)
    ex_layers_num = len(cfg.model.entity_model.hidden_dims)
    for i, layer in enumerate(model.model.layers[-ex_layers_num:]):
        for name, module in layer.named_modules():
            if not isinstance(module, cls):
                continue
            lora_module_names.add(f"model.layers.{layers_num-ex_layers_num + i}.{name}")

    return list(lora_module_names)


# def find_lm_head_names(model: GemmaForCausalLM):
#     lora_module_names = set()
#     for i, layer in enumerate(model.lm_head):
#         for name, module in layer.named_modules():
#             if not isinstance(module, cls):
#                 continue
#             lora_module_names.add(f"model.layers.{layers_num-ex_layers_num + i}.{name}")


def set_requires_grad(
    model: GemmaForCausalLM, module_names: List[str] | str, requires_grad=True
):
    if module_names == '*':
        for name, module in model.named_modules():
            for param in module.parameters():
                param.requires_grad = requires_grad

    for name, module in model.named_modules():
        if any([n in name for n in module_names]):
            for param in module.parameters():
                param.requires_grad = requires_grad


def build_tokenizer(cfg: Config) -> GemmaTokenizer:
    if "gemma" in cfg.model.llm_name:
        tokenizer = GemmaTokenizer.from_pretrained(
            cfg.model.llm_name,
            cache_dir=cfg.model.cache_dir,
            token=cfg.model.hf_token,
            padding_side="left",
            local_files_only=True
        )
    elif "llama" in cfg.model.llm_name:
        # LLaMA3 后暂不能使用 LlamaTokenizer 构建
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.llm_name,
            cache_dir=cfg.model.cache_dir,
            token=cfg.model.hf_token,
            padding_side="left",
        )
    else:
        # AutoPeftModelForCausalLM.from_pretrained()
        # tokenizer = GemmaTokenizer.from_pretrained(
        #     cfg.model.llm_name,
        #     cache_dir=cfg.model.cache_dir,
        #     token=cfg.model.hf_token,
        #     padding_side="left",
        # )
        raise ValueError(f"Invalid LLM name: {cfg.model.llm_name}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def build_model(
    cfg: Config, tokenizer: AutoTokenizer, num_new_tokens: int
) -> PeftModel:
    custom_layers = [
        # "lm_head",
        "graph_model",
        "exchange_info_layer",
        "fushion_layer",
        "special_input_emb",
        "llm_to_rel_layer",
        "llm_to_ent_layer",
        "fuse_llm_ent_layer",
        "proj_rel_layer",
        "rel_norm",
        "ent_norm",
    ]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        # bnb_4bit_compute_dtype=torch.float32,
        # llm_int8_threshold=6.0,
        # llm_int8_has_fp16_weight=False,
        llm_int8_skip_modules=custom_layers,
    )
    model: GNNLLM = GNNLLM.from_pretrained(
        cfg.model.llm_name,
        cache_dir=cfg.model.cache_dir,
        token=cfg.model.hf_token,
        quantization_config=bnb_config,
        cfg=cfg,
        local_files_only=True,
        # torch_dtype=torch.float16
    )

    # TODO 不晓得 Gemma 要不要设置 pad
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False  # Gradient checkpointing is used by default but not compatible with caching

    model.init_graph_tokenizer(tokenizer, num_new_tokens)
    model = prepare_model_for_kbit_training(
        model,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # if not cfg.model.use_peft:
    #     set_requires_grad(model, '*', requires_grad=False)
    #     set_requires_grad(model, custom_layers, requires_grad=True)
    #     return model

    if not cfg.model.load_lora:
        # lora_modules = find_all_linear_names(model, cfg)
        # print(model)
        # print(lora_modules)
        # lora_modules = 'model.embed_tokens'
        lora_modules = 'dummy_layer'
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=lora_modules,
            modules_to_save=custom_layers
        )
        peft_model = get_peft_model(model, peft_config)
    else:
    # if load lora model
        peft_config = PeftConfig.from_pretrained(cfg.model.cache_dir)
        peft_model = PeftModel.from_pretrained(model, cfg.model.cache_dir, config=peft_config)
    
    set_requires_grad(peft_model, custom_layers, requires_grad=True)

    # assert embed_tokens is not requires_grad
    peft_model.base_model.model.model.embed_tokens.weight.requires_grad_(False)
    peft_model.print_trainable_parameters()

    # model = Ultra(
    #     rel_model_cfg=cfg.model.relation_model,
    #     entity_model_cfg=cfg.model.entity_model,
    # )
    return peft_model


def build_tokenizer_model(cfg: Config) -> tuple[AutoTokenizer, PeftModel]:
    tokenizer = build_tokenizer(cfg)
    # # TODO add special tokens
    num_new_tokens = SpecialToken.add_tokens(tokenizer)

    model = build_model(cfg, tokenizer, num_new_tokens)
    # model = TestModel()
    # tokenizer = None
    return tokenizer, model
