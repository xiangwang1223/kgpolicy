import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch_geometric as geometric


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    embed CKG and using its embedding to calculate prediction score
    """

    def __init__(self, in_channel, out_channel):
        super(GraphConv, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1 = geometric.nn.GATConv(in_channel, out_channel)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, edge_indices):
        x = self.conv1(x, edge_indices)
        x = self.dropout(x)
        x = F.normalize(x)
        return x


class KGAT(nn.Module):
    def __init__(self, data_config, args_config):
        super(KGAT, self).__init__()
        self.args_config = args_config
        self.data_config = data_config
        self.n_users = data_config["n_users"]
        self.n_items = data_config["n_items"]
        self.n_nodes = data_config["n_nodes"]

        """set input and output channel manually"""
        input_channel = 64
        output_channel = 64
        self.gcn = GraphConv(input_channel, output_channel)

        self.emb_size = args_config.emb_size
        self.regs = eval(args_config.regs)

        self.all_embed = self._init_weight()

    def _init_weight(self):
        all_embed = nn.Parameter(
            torch.FloatTensor(self.n_nodes, self.emb_size), requires_grad=True
        )
        ui = self.n_users + self.n_items

        if self.args_config.pretrain_r:
            nn.init.xavier_uniform_(all_embed)
            all_embed.data[:ui] = self.data_config["all_embed"]
        else:
            nn.init.xavier_uniform_(all_embed)

        return all_embed

    def build_edge(self, adj_matrix):
        """build edges based on adj_matrix"""
        sample_edge = self.args_config.edge_threshold
        edge_matrix = adj_matrix

        n_node = edge_matrix.size(0)
        node_index = (
            torch.arange(n_node, device=edge_matrix.device)
            .unsqueeze(1)
            .repeat(1, sample_edge)
            .flatten()
        )
        neighbor_index = edge_matrix.flatten()
        edges = torch.cat((node_index.unsqueeze(1), neighbor_index.unsqueeze(1)), dim=1)
        return edges

    def forward(self, user, pos_item, neg_item, edges_matrix):
        u_e, pos_e, neg_e = (
            self.all_embed[user],
            self.all_embed[pos_item],
            self.all_embed[neg_item],
        )

        edges = self.build_edge(edges_matrix)
        x = self.all_embed
        gcn_embedding = self.gcn(x, edges.t().contiguous())

        u_e_, pos_e_, neg_e_ = (
            gcn_embedding[user],
            gcn_embedding[pos_item],
            gcn_embedding[neg_item],
        )

        u_e = torch.cat([u_e, u_e_], dim=1)
        pos_e = torch.cat([pos_e, pos_e_], dim=1)
        neg_e = torch.cat([neg_e, neg_e_], dim=1)

        pos_scores = torch.sum(u_e * pos_e, dim=1)
        neg_scores = torch.sum(u_e * neg_e, dim=1)

        # Defining objective function that contains:
        # ... (1) bpr loss
        bpr_loss = torch.log(torch.sigmoid(pos_scores - neg_scores))
        bpr_loss = -torch.mean(bpr_loss)

        # ... (2) emb loss
        reg_loss = self._l2_loss(u_e) + self._l2_loss(pos_e) + self._l2_loss(neg_e)
        reg_loss = self.regs * reg_loss

        loss = bpr_loss + reg_loss

        return loss, bpr_loss, reg_loss

    def get_reward(self, users, pos_items, neg_items):
        u_e = self.all_embed[users]
        pos_e = self.all_embed[pos_items]
        neg_e = self.all_embed[neg_items]

        neg_scores = torch.sum(u_e * neg_e, dim=1)
        ij = torch.sum(neg_e * pos_e, dim=1)
        reward = neg_scores + ij

        return reward

    def _l2_loss(self, t):
        return torch.sum(t ** 2) / 2

    def inference(self, users):
        num_entity = self.n_nodes - self.n_users - self.n_items
        user_embed, item_embed, _ = torch.split(
            self.all_embed, [self.n_users, self.n_items, num_entity], dim=0
        )

        user_embed = user_embed[users]
        prediction = torch.matmul(user_embed, item_embed.t())
        return prediction

    def rank(self, users, items):
        u_e = self.all_embed[users]
        i_e = self.all_embed[items]

        u_e = u_e.unsqueeze(dim=1)
        ranking = torch.sum(u_e * i_e, dim=2)
        ranking = ranking.squeeze()

        return ranking

    def __str__(self):
        return "recommender using KGAT, embedding size {}".format(
            self.args_config.emb_size
        )
