import torch
import torch.nn as nn
import torch.nn.functional as F 

import torch_geometric as geometric 
import networkx as nx

from tqdm import tqdm

class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    Input: embedding matrix for knowledge graph entity and adjacency matrix
    Output: gcn embedding for kg entity
    """
    def __init__(self, in_channel, out_channel, config):
        super(GraphConv, self).__init__()
        self.config = config
        
        self.conv1 = geometric.nn.SAGEConv(in_channel[0], out_channel[0])
        self.conv2 = geometric.nn.SAGEConv(in_channel[1], out_channel[1])

    def forward(self, x, edge_indices, r_index):
        x = self.conv1(x, edge_indices)
        x = F.leaky_relu(x)
        x = F.dropout(x)

        x = self.conv2(x, edge_indices)
        x = F.dropout(x)
        x = F.normalize(x)
        
        return x


class KGPolicy(nn.Module):
    """
    Dynamical negative item sampler based on Knowledge graph
    Input: user, postive item
    Ouput: qualified negative item
    """
    def __init__(self, dis, params, config):
        super(KGPolicy, self).__init__()
        self.params = params
        self.config = config
        self.dis = dis

        in_channel = eval(config.in_channel)
        out_channel = eval(config.out_channel)
        config.num_relations = params["n_relations"]
        self.gcn = GraphConv(in_channel, out_channel, config)

        self.n_entities = params["n_nodes"]
        self.item_range = params["item_range"]
        self.input_channel = in_channel
        self.entity_embedding = self._initialize_weight(self.n_entities, self.input_channel)

    def _initialize_weight(self, n_entities, input_channel):
        """entities includes items and other entities in knowledge graph"""
        if self.config.pretrain_s:
            kg_embedding = self.params["kg_embedding"]
            entity_embedding = nn.Parameter(kg_embedding)
        else:
            entity_embedding = nn.Parameter(torch.FloatTensor(n_entities, input_channel[0]))
            nn.init.xavier_uniform_(entity_embedding)

        if self.config.freeze_s:
            entity_embedding.requires_grad = False
        return entity_embedding

    def forward(self, data_batch, adj_matrix, edge_matrix, train_set):
        users = data_batch["u_id"]
        pos = data_batch["pos_i_id"]
        neg = data_batch["neg_i_id"]

        self.edges = self.build_edge(edge_matrix)
        
        batch_size = users.size(0)
        good_logits = torch.zeros(batch_size, device=users.device)

        """sample candidate negative items based on knowledge graph"""
        one_hop, logits = self.kg_step(pos, users, adj_matrix, step=1)
        good_logits += logits

        candidate_neg, logits = self.kg_step(one_hop, users, adj_matrix, step=2)

        """replace other entities using random negative items"""
        candidate_neg = self.filter_entity(candidate_neg)

        """using discriminator to further choose qualified negative items"""
        good_neg, logits = self.dis_step(self.dis, candidate_neg, users, logits)

        good_neg = self.filter_trainset(good_neg, train_set, neg)

        good_logits += logits
        good_neg = good_neg.unsqueeze(dim=1)
        good_logits = good_logits.unsqueeze(dim=1)

        """repeat above steps to k times"""
        for s in range(self.config.k_step - 1):
            more_logits = torch.zeros(batch_size, device=users.device)
            one_hop, logits = self.kg_step(good_neg.squeeze(), users, adj_matrix, step=1)
            more_logits += logits
            candidate_neg, logits = self.kg_step(one_hop, users, adj_matrix, step=2)
            candidate_neg = self.filter_entity(candidate_neg)
            more_neg, logits = self.dis_step(self.dis, candidate_neg, users, logits)
            more_neg = self.filter_trainset(more_neg, train_set, neg)
            more_logits += logits
            more_neg = more_neg.unsqueeze(dim=1)
            more_logits = more_logits.unsqueeze(dim=1)

            good_neg = torch.cat([good_neg, more_neg], dim=-1)
            good_logits = torch.cat([good_logits, more_logits], dim=-1)
        
        return good_neg, good_logits

    def filter_trainset(self, negs, train_set, random_set):
        in_train = torch.sum(negs.unsqueeze(1)==train_set.long(), dim=1).byte()
        negs[in_train] = random_set[in_train]
        return negs

    def build_edge(self, adj_matrix):
        """build edges based on adj_matrix"""
        sample_edge = self.config.edge_threshold
        edge_matrix = adj_matrix

        n_node = edge_matrix.size(0)
        node_index = torch.arange(n_node, device=edge_matrix.device).unsqueeze(1).repeat(1, sample_edge).flatten()
        neighbor_index = edge_matrix.flatten()
        edges = torch.cat((node_index.unsqueeze(1), neighbor_index.unsqueeze(1)), dim=1)        
        return edges

    def kg_step(self, pos, user, adj_matrix, step):
        x = self.entity_embedding
        edges = self.edges

        """knowledge graph embedding using gcn"""
        gcn_embedding = self.gcn(x, edges.t().contiguous(), edges.t().contiguous().long())

        """use knowledge embedding to decide candidate negative items"""
        u_e = gcn_embedding[user]
        u_e = u_e.unsqueeze(dim=2)
        pos_e = gcn_embedding[pos]
        pos_e = pos_e.unsqueeze(dim=1)
        
        one_hop = adj_matrix[pos]
        i_e = gcn_embedding[one_hop]

        p_entity = pos_e * i_e 
        p = torch.matmul(p_entity, u_e)
        p = p.squeeze()
        logits = F.softmax(p, dim=1) 

        """sample negative items based on logits"""
        batch_size = logits.size(0)
        if step == 1:
            nid = torch.argmax(logits, dim=1, keepdim=True)
        else:
            n = self.config.num_sample
            _, indices = torch.sort(logits)
            nid = indices[:,:n]
        row_id = torch.arange(batch_size, device=logits.device).unsqueeze(1)

        candidate_neg = one_hop[row_id, nid].squeeze()
        candidate_logits = torch.log(logits[row_id, nid]).squeeze()

        return candidate_neg, candidate_logits

    def dis_step(self, dis, negs, users, logits):
        with torch.no_grad():
            ranking = dis.rank(users, negs)

        """get most qualified negative item based on discriminator"""
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




        
