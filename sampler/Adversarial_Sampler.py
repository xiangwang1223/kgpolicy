import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import scipy.sparse as sp

from utility.test_model import args_config, CKG

class AdvNet(nn.Module):
    """
    Adversarial Net:
    Input: user-item interactions, i.e., <user, pos item> pairs
    Output: hard & informative policy (i.e., negative samples) with corresponding probability
    """
    def __init__(self, data_config, args_config):
        super(AdvNet, self).__init__()

        self.args_config = args_config

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.emb_size = args_config.emb_size
        self.regs = args_config.regs

        self.policy_type = args_config.policy_type

        self.all_embed = self._init_weight()
        self.sp_adj = self._generate_sp_adj()

    def _init_weight(self):

        all_embed = nn.Parameter(torch.FloatTensor(self.n_users + self.n_items, self.emb_size))
        nn.init.xavier_normal_(all_embed)

        # user_embed = nn.Parameter(torch.FloatTensor(self.n_users, self.emb_size))
        # item_embed = nn.Parameter(torch.FloatTensor(self.n_items, self.emb_size))
        # nn.init.xavier_normal_(user_embed)
        # nn.init.xavier_normal_(item_embed)
        #
        # all_weight['all_embed'] = nn.Parameter(torch.cat((user_embed, item_embed), 0))

        return all_embed

    def forward(self, data_batch):
        selected_neg_items, selected_neg_prob = self.neg_sampler(data_batch)
        return selected_neg_items, selected_neg_prob

    def _generate_sp_adj(self):
        train_data = CKG.train_data
        rows = list(train_data[:, 0])
        cols = list(train_data[:, 1])
        vals = [1.] * len(rows)

        return sp.coo_matrix((vals, (rows, cols)), shape=(self.n_users + self.n_items, self.n_users + self.n_items))

    def neg_sampler(self, data_batch):
        def _get_sparse_train_mask(s, idx):
            tmp = s.copy().tolil()
            try:
                tmp = tmp[idx.cpu().numpy()]
            except Exception:
                print(idx)
            return tmp.tocoo()

        def _sparse_dense_mul(sp_coo, ds_torch):
            rows = sp_coo.row
            cols = sp_coo.col
            vals = sp_coo.data
            # get values from relevant entries of dense matrix
            ds_vals = ds_torch[rows, cols]
            return torch.sparse.FloatTensor(torch.LongTensor([rows, cols]), vals * ds_vals, ds_torch.size).to_dense()

        def _masking(ds_torch, train_mask):
            rows = train_mask[0]
            cols = train_mask[1]

            # get values from relevant entries of dense matrix
            ds_vals = ds_torch[rows, cols]
            return torch.sparse.FloatTensor(torch.LongTensor([rows, cols]), ds_vals, ds_torch.size).to_dense()

        user = data_batch['u_id']
        pos_item = data_batch['pos_i_id']
        train_mask = data_batch['train_mask']

        u_e = self.all_embed[user]
        pos_e = self.all_embed[pos_item]
        all_e = self.all_embed

        # # get the mask for positive items appearing in the training set.
        # sp_mask = _get_sparse_train_mask(self.sp_adj, idx=user)

        if self.policy_type == 'uj':
            # ... (1) consider only user preference on item j; the larger, more likely to be selected.
            policy_prob = torch.matmul(u_e, all_e.t())
        elif self.policy_type == 'uij':
            # ... (2) consider user preference on item j, as well as similarity between item i and j.
            policy_prob = torch.matmul(u_e + pos_e, all_e.t())
        else:
            # ... (1) by default set as 'uij'
            policy_prob = torch.matmul(u_e + pos_e, all_e.t())

        # use softmax to calculate sampling probability.
        # policy_prob = _sparse_dense_mul(sp_mask, policy_prob)
        sp_mask = _masking(ds_torch=policy_prob, train_mask=train_mask)
        policy_prob[policy_prob == 0] = -float("inf")
        policy_prob = F.softmax(policy_prob, dim=1)

        # select one negative sample, based on the sampling probability.
        policy_sampler = Categorical(policy_prob)
        selected_neg_items = policy_sampler.sample()
        raw_idx = range(u_e.size(0))
        selected_neg_prob = policy_prob[raw_idx, selected_neg_items]

        return selected_neg_items, selected_neg_prob