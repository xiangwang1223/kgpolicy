import torch
import torch.nn as nn
import torch.nn.functional as F 

import torch_geometric as geometric 
import networkx as nx


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    Input: embedding matrix for knowledge graph entity and adjacency matrix
    Output: gcn embedding for kg entity
    """
    def __init__(self, in_channel, out_channel):
        super(GraphConv, self).__init__()
        in_channel1, in_channel2 = in_channel[0], in_channel[1]
        out_channel1, out_channel2 = out_channel[0], out_channel[1]

        self.conv1 = geometric.nn.GCNConv(in_channel1, out_channel1)
        self.conv2 = geometric.nn.GCNConv(in_channel2, out_channel2)

        self.batch_norm1 = nn.BatchNorm1d(out_channel1)
        self.batch_norm2 = nn.BatchNorm1d(out_channel2)

    def forward(self, x, edge_indices):
        x = self.conv1(x, edge_indices)
        x = F.leaky_relu(x)
        x = self.batch_norm1(x)
        x = F.dropout(x)

        x = self.conv2(x, edge_indices)
        x = self.batch_norm2(x)
        return x


class KGPolicy(nn.Module):
    def __init__(self, dis, params, config):
        super(KGPolicy, self).__init__()
        self.params = params
        self.config = config
        self.dis = dis

        in_channel = eval(config.in_channel)
        out_channel = eval(config.out_channel)
        self.gcn = GraphConv(in_channel, out_channel)

        self.edges = params["edges"]
        self.n_entities = params["n_nodes"]
        self.item_range = params["item_range"]
        self.input_channel = in_channel[0]
        self._initialize_weight(self.n_entities, self.input_channel)

    def _initialize_weight(self, n_entities, input_channel):
        self.entity_embedding = nn.Parameter(torch.FloatTensor(n_entities, input_channel))
        nn.init.xavier_uniform_(self.entity_embedding)

    def forward(self, data_batch, adj_matrix):
        users = data_batch["u_id"]
        pos = data_batch["pos_i_id"]

        one_hop, _ = self.kg_step(pos, users, adj_matrix, step=1)
        candidate_neg, logits = self.kg_step(one_hop, users, adj_matrix, step=2)

        candidate_neg = self.filter_entity(candidate_neg)

        good_neg, good_logits = self.dis_step(self.dis, candidate_neg, users, logits)
        return good_neg, good_logits

    def kg_step(self, pos, user, adj_matrix, step):
        x = self.entity_embedding
        edges = self.edges
        """knowledge graph embedding using gcn"""

        gcn_embedding = self.gcn(x, edges.t().contiguous())

        """use knowledge embedding to decide candidate negative items"""
        u_e = gcn_embedding[user]
        pos_e = gcn_embedding[pos]

        one_hop = adj_matrix[pos]
        i_e = gcn_embedding[one_hop]

        pos_e = pos_e.unsqueeze(dim=1)
        p_entity = pos_e * i_e 

        u_e = u_e.unsqueeze(dim=2)
        p = torch.matmul(p_entity, u_e)
        p = p.squeeze()

        logits = F.softmax(p, dim=1)

        batch_size = logits.size(0)
        if step == 1:
            nid = torch.multinomial(logits, num_samples=1)
        else:
            nid = torch.multinomial(logits, num_samples=self.config.num_sample)
        row_id = torch.arange(batch_size, device=logits.device).unsqueeze(1)

        candidate_neg = one_hop[row_id, nid]
        candidate_neg = candidate_neg.squeeze()

        candidate_logits = logits[row_id, nid]
        candidate_logits = candidate_logits.squeeze()

        return candidate_neg, candidate_logits


    def dis_step(self, dis, negs, users, logits):
        with torch.no_grad():
            ranking = dis.rank(users, negs)
        
        indices = torch.argmax(ranking, dim=1)

        batch_size = negs.size(0)
        row_id = torch.arange(batch_size, device=negs.device).unsqueeze(1)
        indices = indices.unsqueeze(1)

        good_neg = negs[row_id, indices].squeeze()
        goog_logits = logits[row_id, indices].squeeze()

        return good_neg, goog_logits

    def filter_entity(self, neg):
        random_neg = torch.randint(int(self.item_range[0]), int(self.item_range[1] + 1), neg.size(), device=neg.device)
        neg[neg > self.item_range[1]] = random_neg[neg > self.item_range[1]]
        neg[neg < self.item_range[0]] = random_neg[neg < self.item_range[0]]

        return neg




        