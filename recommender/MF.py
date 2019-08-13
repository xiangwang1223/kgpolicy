import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MF(nn.Module):
    def __init__(self, data_config, args_config):
        super(MF, self).__init__()
        self.args_config = args_config
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.emb_size = args_config.emb_size
        self.regs = eval(args_config.regs)

        self.reward_type = args_config.reward_type

        self.all_embed = self._init_weight()

    def _init_weight(self):
        all_embed = nn.Parameter(torch.FloatTensor(self.n_users + self.n_items, self.emb_size))

        nn.init.xavier_normal_(all_embed)

        return all_embed

    def forward(self, data_batch):
        user = data_batch['u_id']
        pos_item = data_batch['pos_i_id']
        neg_item = data_batch['neg_i_id']

        u_e = self.all_embed[user]
        pos_e = self.all_embed[pos_item]
        neg_e = self.all_embed[neg_item]

        pos_scores = torch.sum(u_e * pos_e, dim=1)
        neg_scores = torch.sum(u_e * neg_e, dim=1)

        # Defining objective function that contains:
        # ... (1) bpr loss
        bpr_loss = torch.log(torch.sigmoid(pos_scores - neg_scores))
        bpr_loss = -torch.mean(bpr_loss)
        # ... (2) emb loss
        emb_loss = self._l2_loss(u_e) + self._l2_loss(pos_e) + self._l2_loss(neg_e)
        emb_loss = self.regs[0] * emb_loss

        loss = bpr_loss + emb_loss

        # Defining reward function as:
        reward = 0.
        # if self.reward_type == 'pure':
        #     # ... (1) consider the value of negative scores; the larger, the better;
        #     reward = -torch.log(torch.sigmoid(-neg_scores))
        # elif self.reward_type == 'prod':
        #     # ... (2) consider additionally the inner product of negative and positive embeddings; the larger, the better;
        #     tmp = torch.sum(pos_e * neg_e, dim=1)
        #     reward = -torch.log(torch.sigmoid(-neg_scores)) + tmp
        # else:
        #     # ... (1) by default set as 'pure'.
        #     reward = -torch.log(torch.sigmoid(-neg_scores))

        return reward, loss, bpr_loss, emb_loss

    def _l2_loss(self, t):
        return torch.mean(torch.sum(t ** 2, dim=1) / 2)

    def inference(self, users, items):
        """
        Used for test, calculate predicting score for each items
        Here we use all items as test.
        """
        u_e = self.all_embed[users]
        i_e = self.all_embed[items]

        prediction = torch.matmul(u_e, i_e.t())

        return prediction

    def __str__(self):
        return "recommender using BPRMF, embedding size {}".format(self.args_config.emb_size)


