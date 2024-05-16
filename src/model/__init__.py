from transformers import AutoConfig, AutoModelForCausalLM

from src.model.type import GNNLLMConfig
from src.model.model import GNNLLM

AutoConfig.register("GNNLLM", GNNLLMConfig)
AutoModelForCausalLM.register(GNNLLMConfig, GNNLLM)