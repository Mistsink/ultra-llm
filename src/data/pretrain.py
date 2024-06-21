import random
import torch
from transformers import AutoTokenizer, GemmaTokenizer, LlamaTokenizer
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.loader import LinkNeighborLoader

from config.config import Config
from src.data.types import (
    CustomData,
    CustomSubData,
    CustomSubDataWithSuperNode,
    PretrainDatasetItemOutput,
    PretrainDatasetOutput,
)
from src.data.special_tokens import SpecialToken
from src.ultra import tasks


class PretrainDataset(Dataset):
    def __init__(self, data: CustomData, tokenizer: GemmaTokenizer, cfg: Config):
        self.data = data
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.prompt_len = cfg.task.prompt_len

    def __len__(self):
        return self.cfg.train.batch_per_epoch * self.cfg.train.batch_size

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

        # tasks.negative_sampling(self.data, triple, 2)
        # 随机采一下别的 entity [暂时不写]

        # 对 h，t，entities 采子图
        entities = torch.cat([triple[:, 0], triple[:, 1]]).unique()
        subg = self.sample_from_edge_index(entities)

        # 采样子图中要预测的 triples，以及对应的负样本
        # cfg task num_mask
        mask_triples = None
        while True:
            try:
                mask_triples = self.mask_edges(subg)
                break
            except:
                pass

        # 将 mask_triples 中的 n_id 转成新的子图中的 n_id, 重新标记 [暂时不写]
        origin_id_to_new_id = {
            ori_id.item(): idx for idx, ori_id in enumerate(subg.n_id)
        }
        mask_triples[:, :, :2] = mask_triples[:, :, :2].apply_(
            lambda x: origin_id_to_new_id[x]
        )

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
        entg_prompt, ent_ids = self.create_prompt(subg, "entity")
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
            limit_nodes=subg.n_id,
        )

        return mask_triples

    @staticmethod
    def collate_fn(batch: list[PretrainDatasetItemOutput]) -> PretrainDatasetOutput:
        mask_triples = torch.stack([i.mask_triples for i in batch])
        data = Batch.from_data_list([i.data for i in batch])
        data_rel = Batch.from_data_list([i.data.relation_graph for i in batch])
        data.relation_graph = data_rel
        ent_prompt = torch.stack([i.ent_prompt for i in batch])
        rel_prompt = torch.stack([i.rel_prompt for i in batch])
        # idx ranges in prompt
        rel_begin_idx = torch.tensor([i.rel_begin_idx for i in batch])
        rel_end_idx = torch.tensor([i.rel_end_idx for i in batch])
        rel_ranges = [i.rel_ranges for i in batch]

        ent_begin_idx = torch.tensor([i.ent_begin_idx for i in batch])
        ent_end_idx = torch.tensor([i.ent_end_idx for i in batch])
        ent_ranges = [i.ent_ranges for i in batch]

        labels, id_text_maps = [], []
        for i in batch:
            if i._labels is not None:
                labels.append(i._labels)
            if i._id_text_map is not None:
                id_text_maps.append(i._id_text_map)

        if len(labels) == 0:
            labels = None
        else:
            labels = torch.stack(labels)

        return PretrainDatasetOutput(
            mask_triples=mask_triples,
            data=data,
            data_rel=data_rel,
            ent_prompt=ent_prompt,
            rel_prompt=rel_prompt,
            rel_begin_idx=rel_begin_idx,
            rel_end_idx=rel_end_idx,
            rel_ranges=rel_ranges,
            ent_begin_idx=ent_begin_idx,
            ent_end_idx=ent_end_idx,
            ent_ranges=ent_ranges,
            _labels=labels,
            _id_text_maps=id_text_maps,
        )

    def sample_from_edge_index(
        self, entities: torch.Tensor, return_khop_map: bool = False
    ) -> CustomSubData:
        """
        edge_index: 2 x n or n
        edge_index 可以不存在，会从这两个点开始采样
        # 获取原始 edge_index: sub_g.n_id[sub_g.edge_index]
        # 额外的字段: n_id, e_id, Optional[src_index, dst_pos_index, dst_neg_index]
        #   edge_label_index: 是指从 n_id 的指定节点开始采样
        """
        # negative_sampler = NegativeSampling(mode="triplet", amount=1)
        negative_sampler = None

        if entities.shape[0] % 2 != 0:
            entities = torch.cat([entities, torch.tensor([entities[-1]])])
        # edge_index 为 2 x n 的 tensor
        edge_index = entities.view(2, -1)

        loader = LinkNeighborLoader(
            data=self.data,
            num_neighbors=self.cfg.task.num_neighbors,
            edge_label_index=edge_index,
            subgraph_type="directional",
            disjoint=False,  # TODO 待测试
            neg_sampling=negative_sampler,
            batch_size=edge_index.shape[1],
        )

        # batch size 设为 edge_index.shape[1]，即仅需要一次 iter 即可采样完所有 edge_index
        sub_g: CustomSubData = next(iter(loader))

        # sub_g.n_id 前面几个都是出发点，所以 label 大多都是 0, 1...，最好需要打乱这种偏好
        num_nodes = sub_g.n_id.size(0)
        permuted_indices = torch.randperm(num_nodes, device=sub_g.n_id.device)
        shuffled_n_id = sub_g.n_id[permuted_indices]

        old_idx_oriid_map = {
            idx: ori_id for idx, ori_id in enumerate(sub_g.n_id.tolist())
        }
        new_oriid_idx_map = {
            ori_id: idx for idx, ori_id in enumerate(shuffled_n_id.tolist())
        }

        def _replace(x):
            return new_oriid_idx_map[old_idx_oriid_map[x]]

        sub_g.edge_index = sub_g.edge_index.apply_(_replace)
        sub_g.n_id = shuffled_n_id

        # 过滤掉 逆向 的 edge
        target_index = sub_g.e_id[sub_g.e_id < sub_g.target_edge_type.shape[0]]
        sub_g.target_edge_index = sub_g.target_edge_index[:, target_index]
        sub_g.target_edge_type = sub_g.target_edge_type[target_index]

        if return_khop_map:
            neighbors_map = {}
            for entity in entities.unique():
                new_id = new_oriid_idx_map[entity.item()]
                neighbors_map[new_id] = self.get_neighbors(
                    new_id, sub_g.edge_index, len(self.cfg.task.num_neighbors)
                )
            return sub_g, {
                "oriid_idx_map": new_oriid_idx_map,
                "neighbors_map": neighbors_map,
            }
        return sub_g

    @staticmethod
    def get_neighbors(node_id, edge_index, k):
        all_nodes = set()
        neighbors = []
        # 存储当前跳数的节点
        current_nodes = {node_id}

        # 遍历每一跳，直到达到最大跳数
        for hop in range(k):
            # 存储下一跳的节点
            next_nodes = set()

            # 对于当前跳数的每个节点，查找其邻居节点
            for current_node in current_nodes:
                # 找到所有与当前节点相连接的节点
                # 正向
                connected_nodes = edge_index[1, edge_index[0] == current_node].tolist()
                next_nodes.update(connected_nodes)
                # 逆向
                connected_nodes = edge_index[0, edge_index[1] == current_node].tolist()
                next_nodes.update(connected_nodes)

            # 排除已经存在于当前节点集合中的节点，以避免重复添加
            next_nodes -= all_nodes

            # 将下一跳节点添加到邻居节点集合中
            all_nodes.update(next_nodes)
            neighbors.append(list(next_nodes))

            # 更新当前跳数节点集合
            current_nodes = next_nodes

        return neighbors

    def insert_super_node(
        self, g: CustomSubDataWithSuperNode
    ) -> CustomSubDataWithSuperNode:
        s_n_id = g.num_nodes
        s_edge_index = torch.tensor(
            [[i, s_n_id] for i in range(g.num_nodes)]
            + [[s_n_id, i] for i in range(g.num_nodes)],
            dtype=torch.long,
        ).t()
        s_edge_type = torch.tensor(
            [g.num_edge_types] * g.num_nodes + [g.num_edge_types + 1] * g.num_nodes,
            dtype=torch.long,
        )

        new_edge_index = torch.cat([g.edge_index, s_edge_index], dim=1)
        new_edge_type = torch.cat([g.edge_type, s_edge_type], dim=0)

        g.super_node_id = s_n_id
        g.super_edge_type = torch.tensor(
            [g.num_edge_types, g.num_edge_types + 1], dtype=torch.long
        )
        g.begin_super_edge_index = g.edge_index.shape[1]

        g.edge_index = new_edge_index
        g.edge_type = new_edge_type

        g.num_nodes = g.super_node_id + 1
        g.num_edge_types = g.super_edge_type.max().item() + 1
        return g

    def create_prompt(
        self, g: Data, mode: str, return_text_map=False
    ) -> tuple[str, list[int]]:
        """
        根据 graph 中的 node, 构建 prompt
        """
        assert mode in ["entity", "relation"], "mode should be 'entity' or 'relation'"

        prefix = """Below, I will provide a description document of a Knowledge Graph (KG), showcasing the text information for some nodes within this KG. It will be presented in a specific format:
Example:
<graph-begin> [ENTITY 1] Description of entity node 1 [ENTITY 2] Description of entity node 2 <graph-end>
<graph-begin> [RELATION 1] Description of relation node 1[RELATION 2] Description of relation node 2 <graph-end>
(1) Each KG information will start with <graph-begin> and end with <graph-end>;
(2) Nodes are identified by two types: [ENTITY XX] and [RELATION YY]. "ENTITY" indicates that the node is an entity node, and "RELATION" indicates that the node is a relation node, where XX and YY represent their respective numbers.
(3) The graph is divided into two types: entity graph, where all nodes are ENTITY nodes, and relation graph, where all nodes are RELATION nodes.

Next, I will provide an actual description of a KG:
"""
        text_map = {}

        items: list[tuple[str, int]] = []
        if mode == "entity":
            descs = self.data.text_data.ent_desc
            # entity id 不会因为 构建逆向边 而增加
            # 这里使用 子图中的id写入 prompt，而不是全图的id
            for new_id, i in enumerate(g.n_id.cpu().tolist()):
                desc = descs[i][0]
                items.append((f"[ENTITY {new_id}] {desc}", new_id))
                text_map[new_id] = desc
        else:
            descs = self.data.text_data.rel_desc
            # relation id 会因为 构建逆向边 而增加 2 倍
            # g 也是 relation graph
            assert (
                g.num_nodes % 2 == 0
            ), "relation graph should have even number of nodes"
            for i in range(g.num_nodes // 2):
                desc = descs[i][0]
                items.append((f"[RELATION {i}] {desc}", i))
                text_map[new_id] = desc

        # shuffle items
        random.shuffle(items)
        prompt = (
            prefix
            + SpecialToken.G_BEGIN.value
            + " "
            + " ".join([i[0] for i in items])
            + " "
            + SpecialToken.G_END.value
        )
        node_ids = [i[1] for i in items]

        if return_text_map:
            return prompt, node_ids, text_map

        return prompt, node_ids

    def tokenize(self, texts: list[str], truncate: bool = True) -> torch.Tensor:
        input_ids = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            max_length=self.prompt_len,  #  Gemma supports up to 8k tokens
            truncation=truncate,
        )["input_ids"]

        # TODO 若被 trucation，需要修整
        return input_ids

    def record_node_idx_range(
        self,
        rel_ids: list[int],
        prompt_ids: torch.Tensor,
        mask_triples: torch.Tensor,
        ent_ids: list[int],
    ):
        """
        rel_ids: 按 prompt 中出现顺序的 rel-id
        prompt_ids: 2 x 8k, 0 为 rel prompt, 1 为 ent prompt
        mask_triples: tris x 1+num_neg x 3
        记录各 mask 的节点、负样本在 g、prompt 中的位置，方便在 model encode 后收集特征
        :return: all rel node, masked entity, g-begin, g-end
        rel-range, ent-range 均为稀疏张量sparse_coo_tensor, 读时需要使用 [idx][0] 得到值
            # 准备索引和值
            ids = [[3, 5], [0, 0]]
            vals = [[1, 2], [4, 7]]

            # 创建稀疏张量
            sparse_tensor = torch.sparse_coo_tensor(ids, vals, dtype=torch.float32)

            # 访问指定ID对应的值
            id_to_access = 3
            print(f"ID {id_to_access} 对应的值为: {sparse_tensor[id_to_access][0]}")
        """
        # for rel node range
        g_end_id = self.tokenizer(SpecialToken.G_END.value)["input_ids"][1]
        prompt_ids[:, -1] = g_end_id
        tokens = self.tokenizer.convert_ids_to_tokens(prompt_ids[0])
        prefixs = self.tokenizer.tokenize(" [RELATION")
        rel_ranges, rel_begin_idx, rel_end_idx = self.ranges_from_ids(
            tokens, prefixs, rel_ids
        )

        # for ent node range
        tokens = self.tokenizer.convert_ids_to_tokens(prompt_ids[1])
        prefixs = self.tokenizer.tokenize(" [ENTITY")
        ent_ranges, ent_begin_idx, ent_end_idx = self.ranges_from_ids(
            tokens, prefixs, ent_ids
        )
        """ranges = []
        entities = mask_triples[:, :, :2].unique().cpu().tolist()
        ent_desc_prefixs = [" [ENTITY"]
        ent_desc_prefixs.extend([f" [ENTITY {i}]" for i in entities])
        desc_prefix = self.tokenizer.tokenize(ent_desc_prefixs[0])
        g_begin_token = self.tokenizer.tokenize(SpecialToken.G_BEGIN.value)
        g_end_token = self.tokenizer.tokenize(SpecialToken.G_END.value)

        # 开始查找
        g_begin_idx = find_subsequence_in_list(tokens, g_begin_token, 4)
        for ent_desc_prefix in ent_desc_prefixs[1:]:
            _prefix = self.tokenizer.tokenize(ent_desc_prefix)
            b_idx = find_subsequence_in_list(tokens, _prefix, start_index=g_begin_idx)
            assert b_idx != -1, f"can't find {ent_desc_prefix} in prompt"
            e_idx = find_subsequence_in_list(
                tokens, desc_prefix, start_index=b_idx + len(_prefix)
            )
            if e_idx == -1:
                g_end_idx = find_subsequence_in_list(
                    tokens, g_end_token, 1, start_index=b_idx + len(_prefix)
                )
                assert g_end_idx != -1, f"can't find g_end in prompt"
                e_idx = g_end_idx
            ranges.append((b_idx, e_idx - 1))

        indices = [entities, [0] * len(entities)]
        ent_ranges = torch.sparse_coo_tensor(
            indices,
            ranges,
        )
        ent_begin_idx = g_begin_idx
        ent_end_idx = g_end_idx"""

        return (
            rel_ranges,
            ent_ranges,
            rel_begin_idx,
            rel_end_idx,
            ent_begin_idx,
            ent_end_idx,
            prompt_ids,
        )

    def ranges_from_ids(self, tokens: list[str], prefixs: list[str], ids: list[int]):
        ranges: list[tuple[int, int]] = []

        special_token_const_num = 4
        _g_begin_num, g_begin_idx, g_end_idx = 0, 0, 0
        last_b_idx = -1
        for i, token in enumerate(tokens):
            # 先找到 <graph-begin>
            if (
                token != SpecialToken.G_BEGIN.value
                and _g_begin_num < special_token_const_num
            ):
                continue
            if token == SpecialToken.G_BEGIN.value:
                _g_begin_num += 1
                if _g_begin_num == special_token_const_num:
                    g_begin_idx = i
                continue
            if (
                token == SpecialToken.G_END.value
                and _g_begin_num == special_token_const_num
            ):
                g_end_idx = i
                if last_b_idx != -1:
                    ranges.append((last_b_idx, i - 1))
                break

            # ' [RELATION 123]' -> _[, RELATION, _, 1, 2, 3, ]
            if token == prefixs[0] and tokens[i + 1] == prefixs[1]:
                if last_b_idx == -1:
                    last_b_idx = i
                else:
                    ranges.append((last_b_idx, i - 1))
                    last_b_idx = i

        ranges.extend([(-1, -1)] * (len(ids) - len(ranges)))
        assert len(ranges) == len(ids)

        indices = [ids, [0] * len(ids)]
        rel_ranges = torch.sparse_coo_tensor(
            indices,
            ranges,
        )
        rel_begin_idx = g_begin_idx
        rel_end_idx = g_end_idx

        return rel_ranges, rel_begin_idx, rel_end_idx
