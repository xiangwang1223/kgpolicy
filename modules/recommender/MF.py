import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MF(nn.Module):
    def __init__(self, data_config, args_config):
        super(MF, self).__init__()
        self.args_config = args_config
        self.data_config = data_config
        self.n_users = data_config["n_users"]
        self.n_items = data_config["n_items"]

        self.emb_size = args_config.emb_size
        self.regs = eval(args_config.regs)

        self.all_embed = self._init_weight()

    def _init_weight(self):
        all_embed = nn.Parameter(
            torch.FloatTensor(self.n_users + self.n_items, self.emb_size)
        )

        if self.args_config.pretrain_r:
            all_embed.data = self.data_config["all_embed"]
        else:
            nn.init.xavier_uniform_(all_embed)

        return all_embed

    def forward(self, user, pos_item, neg_item):

        u_e = self.all_embed[user]
        pos_e = self.all_embed[pos_item]
        neg_e = self.all_embed[neg_item]

        reg_loss = self._l2_loss(u_e) + self._l2_loss(pos_e) + self._l2_loss(neg_e)
        reg_loss = self.regs * reg_loss

        pos_scores = torch.sum(u_e * pos_e, dim=1)
        neg_scores = torch.sum(u_e * neg_e, dim=1)

        bpr_loss = torch.log(torch.sigmoid(pos_scores - neg_scores))
        bpr_loss = -torch.mean(bpr_loss)

        return bpr_loss, reg_loss

    def get_reward(self, user, pos_item, neg_item):
        u_e = self.all_embed[user]
        pos_e = self.all_embed[pos_item]
        neg_e = self.all_embed[neg_item]

        neg_scores = torch.sum(u_e * neg_e, dim=-1)
        ij = torch.sum(neg_e * pos_e, dim=-1)

        reward = neg_scores + ij

        return reward

    @staticmethod
    def _l2_loss(t):
        return torch.sum(t ** 2) / 2

    def inference(self, users):
        user_embed, item_embed = torch.split(
            self.all_embed, [self.n_users, self.n_items], dim=0
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
        return "recommender using BPRMF, embedding size {}".format(
            self.args_config.emb_size
        )
