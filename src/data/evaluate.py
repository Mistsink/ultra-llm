import torch
from src.data.instruction import LPInstrucDataset
from src.data.types import PretrainDatasetItemOutput
from src.data.pretrain import PretrainDataset
from src.ultra import tasks


class EvaluateDataset(PretrainDataset):

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

        # 对 h，t，entities 采子图
        entities = torch.cat([triple[:, 0], triple[:, 1]]).unique()
        subg = self.sample_from_edge_index(entities)

        #   检查是不是 0 项 pos，其他项 neg
        # mask_triples: tris x 1+num_neg x 3
        mask_triples = tasks.negative_sampling(
            self.data,  # 这里使用完整图来筛选负样本
            triple,
            self.cfg.task.num_negative,
            strict=self.cfg.task.strict_negative,
            limit_nodes=subg.n_id,
        )
        # 将 mask_triples 中的 n_id 转成新的子图中的 n_id, 重新标记
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
            prompt_ids
        ) = self.record_node_idx_range(rel_ids, prompt_ids, mask_triples, ent_ids)

        input_ids, labels = LPInstrucDataset.get_labels(mask_triples[0], self.tokenizer, max_length=self.cfg.task.instruct_len)

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
            _labels=labels
        )


if __name__ == "__main__":
    e_data = EvaluateDataset()
