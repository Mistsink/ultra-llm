from transformers import GemmaConfig, LlamaConfig


class GNNLLMConfig(LlamaConfig):
    model_type = "GNNLLM"