# class GNNLLMOutput(CausalLMOutputWithPast):
#     ent_emb: Optional[torch.Tensor] = None    1, num_nodes + 1, 2048
#     rel_emb: Optional[torch.Tensor] = None    1, num_rels, 2048


# mask_triples: tris x num_neg+1 x 3
# 对每一个 mask_triple ,构建预测的 prompt:
# 

import torch
from torch.utils.data import Dataset
from transformers import GemmaTokenizer

class LPInstrucDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        