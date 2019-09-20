import numpy as np
import torch
from torch.utils.data import Dataset
import random
import scipy.sparse as sp

from utility.test_model import args_config, CKG

from time import time


class Train_Generator(Dataset):
    def __init__(self, args_config):
        self.args_config = args_config
        # self.sp_adj = self._generate_sp_adj()

        self.user_dict = CKG.train_user_dict
        self.exist_users = list(CKG.exist_users)
        self.low_item_index = CKG.item_range[0]
        self.high_item_index = CKG.item_range[1]

    def __len__(self):
        return CKG.n_train

    def __getitem__(self, index):
        out_dict = {}

        user_dict = self.user_dict
        # randomly select one user.
        u_id = random.sample(self.exist_users, 1)[0]
        out_dict['u_id'] = u_id

        # randomly select one positive item.
        pos_items = user_dict[u_id]
        n_pos_items = len(user_dict[u_id])

        pos_idx = np.random.randint(low=0, high=n_pos_items, size=1)[0]
        pos_i_id = pos_items[pos_idx]

        out_dict['pos_i_id'] = pos_i_id

        neg_i_id = self.get_random_neg(pos_items, [])
        out_dict['neg_i_id'] = neg_i_id

        neg_i_ids = []
        for i in range(self.args_config.k_neg):
            neg_i_id = self.get_random_neg(pos_items, neg_i_ids)
            neg_i_ids.append(neg_i_id)
        out_dict["neg_i_ids"] = torch.tensor(neg_i_ids)

        return out_dict

    def get_random_neg(self, pos_items, selected_items):
        while True:
            neg_i_id = np.random.randint(low=self.low_item_index, high=self.high_item_index, size=1)[0]

            if neg_i_id not in pos_items and neg_i_id not in selected_items:
                break
        return neg_i_id

    def _generate_sp_adj(self):
        train_data = CKG.train_data
        rows = list(train_data[:, 0])
        cols = list(train_data[:, 1])
        vals = [1.] * len(rows)
        n_all = CKG.n_users + CKG.n_items
        return sp.coo_matrix((vals, (rows, cols)), shape=(n_all, n_all))

    def _generate_sp_mask(self, uid):
        mask_adj = self.sp_adj.tolil()[uid]
        mask_adj = mask_adj.tocoo()
        rows = mask_adj.row
        cols = mask_adj.col
        return (rows, cols)

class Test_Generator(Dataset):
    def __init__(self, args_config):
        self.args_config = args_config
        self.users_to_test = list(CKG.test_user_dict.keys())

    def __len__(self):
        return len(self.users_to_test)

    def __getitem__(self, index):
        batch_data = {}

        u_id = self.users_to_test[index]
        batch_data['u_id'] = u_id

        return batch_data

class Sampler_Generator(Train_Generator):
    def __init__(self, args_config):
        super().__init__(args_config)

    def __len__(self):
        return args_config.s_step*args_config.batch_size
