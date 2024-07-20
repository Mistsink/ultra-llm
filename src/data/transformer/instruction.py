if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.getcwd())


import copy
import random
from typing import List
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import GemmaTokenizer

from src.data.types import InstrucInput, find_subsequence_in_list
from src.data.special_tokens import SpecialToken


class DecoderDataset(Dataset):
    def __init__(
        self,
        mask_triples: torch.Tensor,
        ent_emb: torch.Tensor,
        rel_emb: torch.Tensor,
    ):
        super().__init__()
        self.mask_triples = mask_triples.squeeze(0)
        self.ent_emb = ent_emb.squeeze(0) if ent_emb is not None else None
        self.rel_emb = rel_emb.squeeze(0) if rel_emb is not None else None

    def __len__(self):
        return len(self.mask_triples)

    @staticmethod
    def collate_fn(batch: list[InstrucInput]) -> InstrucInput:
        triples = torch.stack([i.triple for i in batch], dim=0)
        label_ids = torch.tensor([i.label_ids for i in batch])
        embs = pad_sequence([i.embs for i in batch], batch_first=True, padding_value=0)

        return InstrucInput(
            triple=triples, input_ids=None, label_ids=label_ids, embs=embs
        )

    def __getitem__(self, idx) -> InstrucInput:
        triples = self.mask_triples[idx]  # num_neg+1 x 3
        h, t, r = triples[0].tolist()
        pred_tail = torch.all(triples[:, 0] == triples[0, 0])

        negs = triples[1:]

        if pred_tail:
            known_e = h
            target_e = t
            neg_ents = negs[:, 1].view(-1).unique().tolist()
        else:
            known_e = t
            target_e = h
            neg_ents = negs[:, 0].view(-1).unique().tolist()

        # node_embs: h, r, shuffled_t_s
        candidate_ents: List[int] = neg_ents + [target_e]
        random.shuffle(candidate_ents)
        labels = candidate_ents.index(target_e)
        assert (
            labels != -1
        ), f"Cannot find the target entity {target_e} in the candidate entities {candidate_ents}"

        node_embs = torch.cat(
            [
                self.ent_emb[known_e].unsqueeze(0),
                self.rel_emb[r].unsqueeze(0),
                self.ent_emb[candidate_ents],
            ],
            dim=0,
        )  # dim:

        return InstrucInput(
            triple=torch.tensor((h, r, t)),
            input_ids=None,
            label_ids=labels,
            embs=node_embs,
        )

    @staticmethod
    def _generate_prompt(
        h, r, t, neg_ents: list[int], id_text_map: dict[int, str], pred_tail=True
    ) -> tuple[str, list[int]]:
        temp = """In the following, I will provide embedding information for multiple entities in a knowledge graph using a specific format:

Example:
 [ENTITY 1] <emb_1> [ENTITY 2] <emb_2>

Following [ENTITY 1] is its embedding information. 
Below is the embedding information for all entities: """
        ents = neg_ents + [h, t]

        random.shuffle(ents)
        ents_str = f" {SpecialToken.INFO_NODE.value} ".join(
            [f"[ENTITY {ent}]" for ent in ents]
        )
        ents_str = ents_str + " " + SpecialToken.INFO_NODE.value
        info_nodes = copy.deepcopy(ents)

        # random.shuffle(ents)
        # brief_desc = ", ".join(
        #     [f"{i+1}.{id_text_map[ent]}" for i, ent in enumerate(ents)]
        # )

        random.shuffle(ents)
        info_nodes_str = " ".join([f"{SpecialToken.INFO_NODE.value}"] * len(ents))
        # suffix = f"Here is a brief description of each entity (not necessarily in the above order): {brief_desc} . Given a sequence of embedding information: {info_nodes_str} , the corresponding IDs and their brief descriptions in order are as follows: "
        suffix = f"Given a sequence of embedding information: {info_nodes_str} , the corresponding IDs are as follows: "

        # prediction_str = ", ".join(
        #     [f"[ENTITY {ent}] {id_text_map[ent]}" for ent in ents]
        # )
        prediction_str = ", ".join([f"[ENTITY {ent}]" for ent in ents])
        prediction_str = prediction_str + "."

        prompt = temp + " " + ents_str + " . " + suffix + prediction_str

        info_nodes = info_nodes + ents

        return prompt, info_nodes

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
