import random
from typing import Optional
import torch
from torch_geometric.data import Data

from src.data.llm_match.instruction_llm_match import LLMMatchInstrucDataset
from src.data.instruction import LPInstrucDataset
from src.data.types import PretrainDatasetItemOutput
from src.data.pretrain import PretrainDataset
from src.ultra import tasks
from src.data.special_tokens import SpecialToken


class EvaluateLLMMatchEmbDataset(PretrainDataset):
    """
    测试 LLM 是否具识别 "ID1: <emb> ID2: <emb>" 中各位置 emb 对应各ID的能力
    """

    def __len__(self):
        fast_test = self.cfg.train.fast_test
        if fast_test != -1:
            return min(self.data.target_edge_index.shape[1], fast_test)
        return self.data.target_edge_index.shape[1]

    def __getitem__(self, idx) -> PretrainDatasetItemOutput:
        idx = idx % self.data.target_edge_index.shape[1]
        triple = (
            torch.cat(
                [
                    self.data.target_edge_index[:, idx],
                    self.data.target_edge_type[idx].unsqueeze(0),
                ]
            )
            .t()
            .view(-1, 3)
        )

        # 1. 完整图中负采样
        while True:
            try:
                # mask_triples: tris x 1+num_neg x 3
                mask_triples = self.mask_edges(self.data, triple)
                break
            except:
                pass

        # 对 h，t，entities 采子图
        entities = torch.cat([mask_triples[:, :, 0], mask_triples[:, :, 1]]).unique()
        subg, map_record = self.sample_from_edge_index(entities, return_khop_map=True)
        oriid_idx_map: dict[int, int] = map_record[
            "oriid_idx_map"
        ]  # 原图中的 id -> new_id(idx)
        neighbors_map: dict[int, list[list[int]]] = map_record[
            "neighbors_map"
        ]  # new_id(idx) -> neighbors[]

        # 将 mask_triples 中的 n_id 转成新的子图中的 n_id, 重新标记
        mask_triples[:, :, :2] = mask_triples[:, :, :2].apply_(
            lambda x: oriid_idx_map[x]
        )
        entities = torch.cat([mask_triples[:, :, 0], mask_triples[:, :, 1]]).unique()

        ht_id = [oriid_idx_map[triple[0, 0].item()], oriid_idx_map[triple[0, 1].item()]]
        neg_t_ids = list(set(entities.tolist()) - set(ht_id))

        # TODO FIXME 随机 mask 一下文本信息或者结构信息 [暂时不写]

        # 创建 super_node, 并创建一条特殊的 edge_type
        subg = self.insert_super_node(subg)

        # 构建 relation graph
        #   a. 从 subg 中提取出所有的 relation graph [暂时不写]
        #       subg = tasks.build_relation_graph(subg)
        #   b. 使用完整的 graph 的 relation graph，这样不会丢失 query rel，否则可能会缺失某些 rel
        assert self.data.relation_graph is not None, "relation graph is None"
        relg = self.data.relation_graph

        # 编写 rel_g, ent_g 的 prompt
        #   a. 各子图单独一个 prompt [暂时不写]
        #   b. 子图 batch 成一个 g，直接 concat 三元组即可，共享同一个 prompt
        relg_prompt, rel_ids = self.create_prompt(relg, "relation")

        pred_tail = torch.all(mask_triples[0, :, 0] == mask_triples[0, 0, 0])
        entg_prompt, ent_ids = self.create_prompt(
            subg,
            "entity",
            ht_id=ht_id,
            neg_t_ids=neg_t_ids,
            neighbors_map=neighbors_map,
            pref_tail=pred_tail,
        )
        prompt_ids = self.tokenize([relg_prompt, entg_prompt])

        # 记录各 mask 的节点、负样本在 g、prompt 中的位置，方便在 model encode 后收集特征
        (
            rel_ranges,
            ent_ranges,
            rel_begin_idx,
            rel_end_idx,
            ent_begin_idx,
            ent_end_idx,
            prompt_ids,
        ) = self.record_node_idx_range(rel_ids, prompt_ids, mask_triples, ent_ids)

        # TODO FIXME ent_ranges 中没有 subg.n_id 中所有内容

        input_ids, labels = LLMMatchInstrucDataset.get_labels(
            mask_triples[0],
            self.tokenizer,
            eos_token=self.tokenizer.eos_token,
            max_length=self.cfg.task.instruct_len,
        )

        return PretrainDatasetItemOutput(
            mask_triples=mask_triples,
            data=subg,
            ent_prompt=prompt_ids[1],
            rel_prompt=prompt_ids[0],
            rel_begin_idx=rel_begin_idx,
            rel_end_idx=rel_end_idx,
            rel_ranges=rel_ranges,
            ent_begin_idx=ent_begin_idx,
            ent_end_idx=ent_end_idx,
            ent_ranges=ent_ranges,
            _labels=labels,
        )

    def mask_edges(self, subg, triples=None):
        if triples is None:
            edge_mask = torch.randperm(subg.target_edge_index.shape[1])[
                : self.cfg.task.num_mask
            ]
            # mask_triples: tris x 3
            triples = (
                torch.cat(
                    [
                        subg.target_edge_index[:, edge_mask],
                        subg.target_edge_type[edge_mask].unsqueeze(0),
                    ]
                )
                .t()
                .view(-1, 3)
            )

        #   检查是不是 0 项 pos，其他项 neg
        # mask_triples: tris x 1+num_neg x 3
        mask_triples = tasks.negative_sampling(
            self.data,  # 这里使用完整图来筛选负样本
            triples,
            self.cfg.task.num_negative,
            strict=self.cfg.task.strict_negative,
            limit_nodes=getattr(subg, "n_id", None),
        )

        return mask_triples

    def create_prompt(
        self,
        g: Data,
        mode: str,
        ht_id: Optional[list[int]] = None,
        neg_t_ids: Optional[list[int]] = None,
        neighbors_map: Optional[dict[int, list[list[int]]]] = None,
        pref_tail: Optional[bool] = None,
    ) -> tuple[str, list[int]]:
        """
        根据 graph 中的 node, 构建 prompt
        """
        assert mode in ["entity", "relation"], "mode should be 'entity' or 'relation'"
        if mode == "relation":
            return super().create_prompt(g, mode)

        assert (
            ht_id is not None
            and neg_t_ids is not None
            and neighbors_map is not None
            and pref_tail is not None
        ), "entity mode must input ht_id, neg_t_ids, neighbors_map, pref_tail params in EVAL_STAGE"

        prefix = """Below, I will provide a description document of a Knowledge Graph (KG), showcasing the text information for some nodes within this KG. It will be presented in a specific format:
Example:
<graph-begin> [ENTITY 1] Description of entity node 1 [ENTITY 2] Description of entity node 2 <graph-end>
<graph-begin> [RELATION 1] Description of relation node 1[RELATION 2] Description of relation node 2 <graph-end>
(1) Each KG information will start with <graph-begin> and end with <graph-end>;
(2) Nodes are identified by two types: [ENTITY XX] and [RELATION YY]. "ENTITY" indicates that the node is an entity node, and "RELATION" indicates that the node is a relation node, where XX and YY represent their respective numbers.
(3) The graph is divided into two types: entity graph, where all nodes are ENTITY nodes, and relation graph, where all nodes are RELATION nodes.

Next, I will provide an actual description of a KG:
"""

        id_oriid_map = g.n_id.tolist()
        descs = self.data.text_data.ent_desc

        items: list[list[tuple[str, int]]] = []
        n_records = set()

        if pref_tail:
            known_ht_id = ht_id[0]
            unknown_ht_id = ht_id[1]
        else:
            known_ht_id = ht_id[1]
            unknown_ht_id = ht_id[0]

        # 分层级写入 prompt，每一层可以 shuffle
        # 0 层
        _items: list[tuple[str, int]] = []
        #   h, t
        _items.append(
            (
                f"[ENTITY {known_ht_id}] {descs[id_oriid_map[known_ht_id]][0]}",
                known_ht_id,
            )
        )
        n_records.add(known_ht_id)
        _items.append(
            (
                f"[ENTITY {unknown_ht_id}] {descs[id_oriid_map[unknown_ht_id]][0]}",
                unknown_ht_id,
            )
        )
        n_records.add(unknown_ht_id)
        #   neg_t
        for _id in neg_t_ids:
            if _id in n_records:
                continue
            _items.append((f"[ENTITY {_id}] {descs[id_oriid_map[_id]][0]}", _id))
            n_records.add(_id)
        random.shuffle(_items)
        items.append(_items)

        # i hop
        for i in range(len(neighbors_map[known_ht_id])):
            _items = []
            # known_ht_id[i]
            for _id in neighbors_map[known_ht_id][i]:
                if _id in n_records:
                    continue
                _items.append((f"[ENTITY {_id}] {descs[id_oriid_map[_id]][0]}", _id))
                n_records.add(_id)
            # unknown_ht_id[i]
            for _id in neighbors_map[unknown_ht_id][i]:
                if _id in n_records:
                    continue
                _items.append((f"[ENTITY {_id}] {descs[id_oriid_map[_id]][0]}", _id))
                n_records.add(_id)
            # neg_t[i]
            for neg_t_id in neg_t_ids:
                for _id in neighbors_map[neg_t_id][i]:
                    if _id in n_records:
                        continue
                    _items.append(
                        (f"[ENTITY {_id}] {descs[id_oriid_map[_id]][0]}", _id)
                    )
                    n_records.add(_id)
            random.shuffle(_items)
            items.append(_items)

        assert len(id_oriid_map) == len(n_records), "n_id中有节点未添加到prompt中"

        # flat
        items = [i for _items in items for i in _items]

        return prefix + SpecialToken.G_BEGIN.value + " " + " ".join(
            [i[0] for i in items]
        ) + " " + SpecialToken.G_END.value, [i[1] for i in items]


if __name__ == "__main__":
    e_data = EvaluateDataset()
