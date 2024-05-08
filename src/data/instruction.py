if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.getcwd())

# class GNNLLMOutput(CausalLMOutputWithPast):
#     ent_emb: Optional[torch.Tensor] = None    1, num_nodes + 1, 2048
#     rel_emb: Optional[torch.Tensor] = None    1, num_rels, 2048


# mask_triples: tris x num_neg+1 x 3
# 对每一个 mask_triple ,构建预测的 prompt:
#

import torch
from torch.utils.data import Dataset
from transformers import GemmaTokenizer

from src.data.types import InstrucInput, find_subsequence_in_list
from src.data.special_tokens import SpecialToken


class LPInstrucDataset(Dataset):
    def __init__(
        self,
        mask_triples: torch.Tensor,
        ent_emb: torch.Tensor,
        rel_emb: torch.Tensor,
        tokenizer: GemmaTokenizer,
        max_length=512,
    ):
        super().__init__()
        self.mask_triples = mask_triples
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
        embs = torch.stack([i.embs for i in batch], dim=0)

        return InstrucInput(triple=triples, input_ids=input_ids, label_ids=label_ids, embs=embs)

    def __getitem__(self, idx) -> InstrucInput:
        triples = self.mask_triples[idx]  # num_neg+1 x 3
        h, r, t = triples[0].tolist()

        # TODO debug 查清楚 前面多少是 t_negs
        num_negs = len(triples[1:])
        t_negs = triples[1 : num_negs // 2 + 1]
        h_negs = triples[num_negs // 2 + 1 :]

        t_neg_ents = t_negs[:, 1].view(-1).unique().tolist()
        h_neg_ents = h_negs[:, 0].view(-1).unique().tolist()

        # 生成 prompt
        r_emb = self.rel_emb[r].unsqueeze(0)
        pred_t_prompt, node_ids = self._generate_prompt(
            h, r, t, t_neg_ents, pred_tail=True
        )
        node_embs = self.ent_emb[node_ids]
        node_embs = torch.cat([node_embs, r_emb], dim=0).unsqueeze(
            0
        )  # 1 x num_nodes+1 x dim
        pred_h_prompt, node_ids = self._generate_prompt(
            h, r, t, h_neg_ents, pred_tail=False
        )
        _node_embs = self.ent_emb[node_ids]
        _node_embs = torch.cat([_node_embs, r_emb], dim=0).unsqueeze(
            0
        )  # 1 x num_nodes+1 x dim
        node_embs = torch.cat([node_embs, _node_embs], dim=0).unsqueeze(
            0
        )  # 1 x 2 x num_nodes+1 x dim

        # tokenization & labeling
        pred_t_input_ids, pred_t_labels = self.tokenize(
            pred_t_prompt, len(t_neg_ents) + 3
        )
        pred_h_input_ids, pred_h_labels = self.tokenize(
            pred_h_prompt, len(h_neg_ents) + 3
        )

        input_ids = torch.cat(
            [pred_t_input_ids.unsqueeze(0), pred_h_input_ids.unsqueeze(0)], dim=0
        )
        labels = torch.cat(
            [pred_t_labels.unsqueeze(0), pred_h_labels.unsqueeze(0)], dim=0
        )

        return InstrucInput(
            triple=torch.tensor((h, r, t)),
            input_ids=input_ids,
            label_ids=labels,
            embs=node_embs,
        )

    def _generate_prompt(
        self, h, r, t, neg_ents: list[int], pred_tail=True
    ) -> tuple[str, list[int]]:
        """
                Template:
                In the following, I will provide embedding information for multiple entities in a knowledge graph using a specific format:

        Example:
         [ENTITY 1] <emb> [ENTITY 2] <emb>

        Following [ENTITY 1] is its embedding information. Please complete the triplet task, where we know a relation [RELATION r] and either the head entity [ENTITY h] or the tail entity [ENTITY t]. Predict the missing tail entity [ENTITY t] (or head entity [ENTITY h]).

        Below is the embedding information for all candidate tail entities: [ENTITY t1] <> [ENTITY t2] <> [ENTITY t3] <> .

        We know the embedding information for a missing triplet's head entity: [ENTITY h_id] <>, and the embedding information for its relation: [RELATION r_id]<>. The missing tail entity is: [ENTITY t_id].
        """
        temp = """In the following, I will provide embedding information for multiple entities in a knowledge graph using a specific format:

Example:
 [ENTITY 1] <emb> [ENTITY 2] <emb>

Following [ENTITY 1] is its embedding information. Please complete the triplet task, where we know a relation [RELATION r] and either the head entity [ENTITY h] or the tail entity [ENTITY t]. Predict the missing tail entity [ENTITY t] (or head entity [ENTITY h]).

Below is the embedding information for all candidate tail entities: """
        if pred_tail:
            ents = neg_ents + [t]
            suffix = f"We know the embedding information for a missing triplet's head entity: [ENTITY {h}] {SpecialToken.INFO_NODE.value}, and the embedding information for its relation: [RELATION {r}] {SpecialToken.INFO_NODE.value}. The missing tail entity is: [ENTITY {t}]."
        else:
            ents = neg_ents + [h]
            suffix = f"We know the embedding information for a missing triplet's tail entity: [ENTITY {t}] {SpecialToken.INFO_NODE.value}, and the embedding information for its relation: [RELATION {r}] {SpecialToken.INFO_NODE.value}. The missing head entity is: [ENTITY {h}]."

        ents_str = f" {SpecialToken.INFO_NODE.value} ".join(
            [f"[ENTITY {ent}]" for ent in ents]
        )
        ents_str = ents_str + " " + SpecialToken.INFO_NODE.value

        prompt = temp + " " + ents_str + " . " + suffix

        return prompt, ents

    def tokenize(
        self, prompt: str, num_info_nodes: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )["input_ids"][0]
        # 初始化 label: 全为 -100
        labels = torch.full_like(input_ids, -100)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        info_node_tokens = self.tokenizer.tokenize(SpecialToken.INFO_NODE.value)
        last_info_node_idx = find_subsequence_in_list(
            tokens, info_node_tokens, num_info_nodes
        )
        assert (
            last_info_node_idx != -1
        ), "Cannot find enough info node tokens in the tokenized prompt."
        label_tokens_prefix = self.tokenizer.tokenize(" [ENTITY")
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
    lp_data = LPInstrucDataset(triples, ent_emb, rel_emb, tokenizer)
    t = lp_data[0]
    print(t)
    pass
