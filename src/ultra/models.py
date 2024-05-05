from typing import Optional
import torch
from torch import nn

from . import tasks, layers
from src.ultra.base_nbfnet import BaseNBFNet


class Ultra(nn.Module):

    def __init__(self, rel_model_cfg, entity_model_cfg):
        # kept that because super Ultra sounds cool
        super(Ultra, self).__init__()

        self.relation_model = RelNBFNet(**rel_model_cfg)
        self.entity_model = EntityNBFNet(**entity_model_cfg)

    def forward(self, data, batch):

        # batch shape: (bs, 1+num_negs, 3)
        # relations are the same all positive and negative triples, so we can extract only one from the first triple among 1+nug_negs
        query_rels = batch[:, 0, 2]
        relation_representations = self.relation_model(
            data.relation_graph, query=query_rels
        )
        score = self.entity_model(data, relation_representations, batch)

        return score


# NBFNet to work on the graph of relations with 4 fundamental interactions
# Doesn't have the final projection MLP from hidden dim -> 1, returns all node representations
# of shape [bs, num_rel, hidden]
class RelNBFNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation=4, **kwargs):
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i],
                    self.dims[i + 1],
                    num_relation,
                    self.dims[0],
                    self.message_func,
                    self.aggregate_func,
                    self.layer_norm,
                    self.activation,
                    dependent=False,
                )
            )

        if self.concat_hidden:
            feature_dim = sum(hidden_dims) + input_dim
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, input_dim),
            )

    def bellmanford(
        self,
        data,
        h_index: Optional[torch.Tensor] = None,
        separate_grad=False,
        boundary: Optional[torch.Tensor] = None,
    ):
        # 原本是指 query_rel 的初始化 emb(全为1), 现在是将 LLM 的输出作为初始化 emb，即 boundary
        if boundary is None:
            # initialize initial nodes (relations of interest in the batch) with all ones
            query = torch.ones(
                h_index.shape[0], self.dims[0], device=h_index.device, dtype=torch.float
            )
            index = h_index.unsqueeze(-1).expand_as(query)
        else:
            query = boundary

        if (
            boundary is None
        ):  # 原本的做法是 query_rel 为1，其他全为0，现在使用 LLM 的输出作为 boundary
            batch_size = len(h_index)
            # initial (boundary) condition - initialize all node states as zeros
            boundary = torch.zeros(
                batch_size, data.num_nodes, self.dims[0], device=h_index.device
            )
            # boundary = torch.zeros(data.num_nodes, *query.shape, device=h_index.device)
            # Indicator function: by the scatter operation we put ones as init features of source (index) nodes
            boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))

        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:
            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(
                layer_input,
                query,
                boundary,
                data.edge_index,
                data.edge_type,
                size,
                edge_weight,
            )
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(
            -1, data.num_nodes, -1
        )  # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
            output = self.mlp(output)
        else:
            output = hiddens[-1]

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, rel_graph, query=None, boundary: Optional[torch.Tensor] = None):

        # message passing and updated node representations (that are in fact relations)
        output = self.bellmanford(rel_graph, h_index=query, boundary=boundary)[
            "node_feature"
        ]  # (batch_size, num_nodes, hidden_dim）

        return output


class EntityNBFNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation=1, **kwargs):

        # dummy num_relation = 1 as we won't use it in the NBFNet layer
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i],
                    self.dims[i + 1],
                    num_relation,
                    self.dims[0],
                    self.message_func,
                    self.aggregate_func,
                    self.layer_norm,
                    self.activation,
                    dependent=False,
                    project_relations=True,
                )
            )

        feature_dim = (
            sum(hidden_dims) if self.concat_hidden else hidden_dims[-1]
        ) + input_dim
        self.mlp = nn.Sequential()
        mlp = []
        for i in range(self.num_mlp_layers - 1):
            mlp.append(nn.Linear(feature_dim, feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)

    def bellmanford(self, data, h_index, r_index, separate_grad=False, boundary: Optional[torch.Tensor]=None):
        if boundary is None:
            batch_size = len(r_index)

            # initialize queries (relation types of the given triples)
            query = self.query[torch.arange(batch_size, device=r_index.device), r_index]
            index = h_index.unsqueeze(-1).expand_as(query)

            # initial (boundary) condition - initialize all node states as zeros
            boundary = torch.zeros(
                batch_size, data.num_nodes, self.dims[0], device=h_index.device
            )
        # by the scatter operation we put query (relation) embeddings as init features of source (index) nodes
            boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        else:
            query = self.query

        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:

            # for visualization
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()

            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(
                layer_input,
                query,  # only for cal batch_size
                boundary,
                data.edge_index,
                data.edge_type,
                size,
                edge_weight,
            )
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(
            -1, data.num_nodes, -1
        )  # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def bellmanford_layer(self, data, h_index, r_index, separate_grad=False, boundary: Optional[torch.Tensor]=None, layer_idx: int=-1):
        assert layer_idx > 0, "layer_idx should be greater than 0 in ent-gnn bellmanford_layer."

        if boundary is None:
            batch_size = len(r_index)

            # initialize queries (relation types of the given triples)
            query = self.query[torch.arange(batch_size, device=r_index.device), r_index]
            index = h_index.unsqueeze(-1).expand_as(query)

            # initial (boundary) condition - initialize all node states as zeros
            boundary = torch.zeros(
                batch_size, data.num_nodes, self.dims[0], device=h_index.device
            )
        # by the scatter operation we put query (relation) embeddings as init features of source (index) nodes
            boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        else:
            query = self.query

        size = (data.num_nodes, data.num_nodes)

        if layer_idx == 0:
            self.hiddens = []
            self.edge_weights = []
            layer_input = boundary
            edge_weight = torch.ones(data.num_edges, device=h_index.device)
        else:
            layer_input = self.hiddens[layer_idx-1]
            edge_weight = self.edge_weights[layer_idx-1]

        layer = self.layers[layer_idx]
        # for visualization
        if separate_grad:
            edge_weight = edge_weight.clone().requires_grad_()
        hidden = layer(
            layer_input,
            query,  # only for cal batch_size
            boundary,
            data.edge_index,
            data.edge_type,
            size,
            edge_weight,
        )
        if self.short_cut and hidden.shape == layer_input.shape:
            # residual connection here
            hidden = hidden + layer_input
        self.hiddens.append(hidden)
        self.edge_weights.append(edge_weight)

        if layer_idx == len(self.layers) - 1:
            # original query (relation type) embeddings
            node_query = query.unsqueeze(1).expand(
                -1, data.num_nodes, -1
            )
            if self.concat_hidden:
                output = torch.cat(self.hiddens + [node_query], dim=-1)
            else:
                output = torch.cat([self.hiddens[-1], node_query], dim=-1)

            return {
                "node_feature": output,
                "edge_weights": self.edge_weights,
            }
        else:
            return {
                "node_feature": hidden,
                "edge_weights": edge_weight,
            }

    """def forward(self, data, relation_representations, batch):
        h_index, t_index, r_index = batch.unbind(-1)

        # initial query representations are those from the relation graph
        self.query = relation_representations

        # initialize relations in each NBFNet layer (with uinque projection internally)
        for layer in self.layers:
            layer.relation = relation_representations

        if self.training:
            # Edge dropout in the training mode
            # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
            # to make NBFNet iteration learn non-trivial paths
            data = self.remove_easy_edges(data, h_index, t_index, r_index)

        shape = h_index.shape
        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index = self.negative_sample_to_tail(
            h_index, t_index, r_index, num_direct_rel=data.num_relations // 2
        )
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        # message passing and updated node representations
        output = self.bellmanford(
            data, h_index[:, 0], r_index[:, 0]
        )  # (num_nodes, batch_size, feature_dim）
        feature = output["node_feature"]
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        # extract representations of tail entities from the updated node states
        feature = feature.gather(
            1, index
        )  # (batch_size, num_negative + 1, feature_dim)

        # probability logit for each tail node in the batch
        # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
        score = self.mlp(feature).squeeze(-1)
        return score.view(shape)"""
    def forward(self, data, relation_representations, batch, boundary: Optional[torch.Tensor]=None, layer_idx: int=-1):
        assert layer_idx > 0, "layer_idx should be greater than 0 in ent-gnn model."

        h_index, t_index, r_index = batch.unbind(-1)

        # initial query representations are those from the relation graph
        self.query = relation_representations

        # initialize relations in each NBFNet layer (with uinque projection internally)
        # for layer in self.layers:
        #     layer.relation = relation_representations
        self.layers[layer_idx].relation = relation_representations

        if self.training:
            # Edge dropout in the training mode
            # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
            # to make NBFNet iteration learn non-trivial paths
            data = self.remove_easy_edges(data, h_index, t_index, r_index)

        # shape = h_index.shape
        # # turn all triples in a batch into a tail prediction mode
        # h_index, t_index, r_index = self.negative_sample_to_tail(
        #     h_index, t_index, r_index, num_direct_rel=data.num_relations // 2
        # )
        # assert (h_index[:, [0]] == h_index).all()
        # assert (r_index[:, [0]] == r_index).all()

        # message passing and updated node representations
        output = self.bellmanford(
            data, h_index[:, 0], r_index[:, 0], boundary=boundary
        )  # (num_nodes, batch_size, feature_dim）
        output = self.bellmanford_layer(
            data, h_index[:, 0], r_index[:, 0], boundary=boundary, layer_idx=layer_idx
        )  # (num_nodes, batch_size, feature_dim）
        feature = output["node_feature"]
        return feature
