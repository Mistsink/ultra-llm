if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.getcwd())


import random
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import GemmaTokenizer

from src.data.types import InstrucInput, find_subsequence_in_list
from src.data.special_tokens import SpecialToken


class LLMMatchInstrucDataset(Dataset):
    def __init__(
        self,
        mask_triples: torch.Tensor,
        ent_emb: torch.Tensor,
        rel_emb: torch.Tensor,
        tokenizer: GemmaTokenizer,
        max_length=512,
    ):
        super().__init__()
        self.mask_triples = mask_triples.squeeze(0)
        self.ent_emb = ent_emb.squeeze(0)
        self.rel_emb = rel_emb.squeeze(0)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.mask_triples)

    @staticmethod
    def collate_fn(batch: list[InstrucInput]) -> InstrucInput:
        triples = torch.stack([i.triple for i in batch], dim=0)
        input_ids = torch.stack([i.input_ids for i in batch], dim=0)
        label_ids = torch.stack([i.label_ids for i in batch], dim=0)
        embs = pad_sequence([i.embs for i in batch], batch_first=True, padding_value=0)

        return InstrucInput(
            triple=triples, input_ids=input_ids, label_ids=label_ids, embs=embs
        )

    @staticmethod
    def get_labels(
        triples: torch.Tensor,
        tokenizer: GemmaTokenizer,
        eos_token: str,
        max_length: int = 512,
    ):
        h, t, r = triples[0].tolist()
        pred_tail = torch.all(triples[:, 0] == triples[0, 0])

        negs = triples[1:]

        if pred_tail:
            neg_ents = negs[:, 1].view(-1).unique().tolist()
        else:
            neg_ents = negs[:, 0].view(-1).unique().tolist()

        # 生成 prompt
        prompt, node_ids = LLMMatchInstrucDataset._generate_prompt(
            h, r, t, neg_ents, pred_tail=pred_tail
        )
        prompt += eos_token

        # tokenization & labeling
        input_ids, labels = LLMMatchInstrucDataset.tokenize(
            prompt, len(neg_ents) + 1, tokenizer, max_length=max_length
        )

        input_ids, labels = input_ids.squeeze(0), labels.squeeze(0)

        return input_ids, labels

    def __getitem__(self, idx) -> InstrucInput:
        triples = self.mask_triples[idx]  # num_neg+1 x 3
        h, t, r = triples[0].tolist()
        pred_tail = torch.all(triples[:, 0] == triples[0, 0])

        negs = triples[1:]

        if pred_tail:
            neg_ents = negs[:, 1].view(-1).unique().tolist()
        else:
            neg_ents = negs[:, 0].view(-1).unique().tolist()

        # 生成 prompt
        r_emb = self.rel_emb[r].unsqueeze(0)
        prompt, node_ids = self._generate_prompt(h, r, t, neg_ents, pred_tail=pred_tail)
        prompt += self.tokenizer.eos_token
        node_embs = self.ent_emb[node_ids]
        # node_embs = torch.cat([node_embs, r_emb], dim=0)  # num_nodes+1 x dim

        # tokenization & labeling
        input_ids, labels = self.tokenize(
            prompt,
            len(neg_ents) + 1,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

        input_ids, labels = input_ids.squeeze(0), labels.squeeze(0)

        return InstrucInput(
            triple=torch.tensor((h, r, t)),
            input_ids=input_ids,
            label_ids=labels,
            embs=node_embs,
        )

    @staticmethod
    def _generate_prompt(
        h, r, t, neg_ents: list[int], pred_tail=True
    ) -> tuple[str, list[int]]:
        temp = """In the following, I will provide embedding information for multiple entities in a knowledge graph using a specific format:

Example:
 [ENTITY 1] <emb> [ENTITY 2] <emb>

Following [ENTITY 1] is its embedding information. Please provide the corresponding entity ID based on the specified embedding information.

Below is the embedding information for all candidate entities: """
        ents = neg_ents


        random.shuffle(ents)
        target_ent = random.choice(ents)
        suffix = f"The embedding of an entity is {SpecialToken.INFO_NODE.value}, its corresponding entity is [ENTITY {target_ent}]."

        ents_str = f" {SpecialToken.INFO_NODE.value} ".join(
            [f"[ENTITY {ent}]" for ent in ents]
        )
        ents_str = ents_str + " " + SpecialToken.INFO_NODE.value

        prompt = temp + " " + ents_str + " . " + suffix

        return prompt, ents + [target_ent]

    @staticmethod
    def tokenize(
        prompt: str, num_info_nodes: int, tokenizer: GemmaTokenizer, max_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )["input_ids"][0]
        # 初始化 label: 全为 -100
        labels = torch.full_like(input_ids, -100)

        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        info_node_tokens = tokenizer.tokenize(SpecialToken.INFO_NODE.value)
        last_info_node_idx = find_subsequence_in_list(
            tokens, info_node_tokens, num_info_nodes
        )
        assert (
            last_info_node_idx != -1
        ), f"Cannot find enough info node tokens in the tokenized prompt.\ntoekns:{tokens[-20:]}\nnum_nodes: {num_info_nodes}"
        label_tokens_prefix = tokenizer.tokenize(" [ENTITY")
        label_idx = find_subsequence_in_list(
            tokens, label_tokens_prefix, start_index=last_info_node_idx
        )
        assert label_idx != -1, "Cannot find the label tokens in the tokenized prompt."

        labels[label_idx + len(label_tokens_prefix) :] = input_ids[
            label_idx + len(label_tokens_prefix) :
        ]

        return input_ids, labels


if __name__ == "__main__":

    triples = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    tokenizer = GemmaTokenizer.from_pretrained(
        "google/gemma-2b", cache_dir="hf_models", padding_side="left"
    )  # padding_side="left"
    n = SpecialToken.add_tokens(tokenizer)

    ent_emb = torch.randn(10, 12)
    rel_emb = torch.randn(10, 12)
    lp_data = LLMMatchInstrucDataset(triples, ent_emb, rel_emb, tokenizer)
    t = lp_data[0]
    print(t)
    pass
