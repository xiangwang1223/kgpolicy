import numpy as np
import torch
from torch.utils.data import Dataset
import random
import scipy.sparse as sp

from time import time


class TrainGenerator(Dataset):
    def __init__(self, args_config, graph):
        self.args_config = args_config
        self.graph = graph

        self.user_dict = graph.train_user_dict
        self.exist_users = list(graph.exist_users)
        self.low_item_index = graph.item_range[0]
        self.high_item_index = graph.item_range[1]

    def __len__(self):
        return self.graph.n_train

    def __getitem__(self, index):
        out_dict = {}

        user_dict = self.user_dict
        # randomly select one user.
        u_id = random.sample(self.exist_users, 1)[0]
        out_dict["u_id"] = u_id

        # randomly select one positive item.
        pos_items = user_dict[u_id]
        n_pos_items = len(user_dict[u_id])

        pos_idx = np.random.randint(low=0, high=n_pos_items, size=1)[0]
        pos_i_id = pos_items[pos_idx]

        out_dict["pos_i_id"] = pos_i_id

        neg_i_id = self.get_random_neg(pos_items, [])
        out_dict["neg_i_id"] = neg_i_id

        return out_dict

    def get_random_neg(self, pos_items, selected_items):
        while True:
            neg_i_id = np.random.randint(
                low=self.low_item_index, high=self.high_item_index, size=1
            )[0]

            if neg_i_id not in pos_items and neg_i_id not in selected_items:
                break
        return neg_i_id


class TestGenerator(Dataset):
    def __init__(self, args_config, graph):
        self.args_config = args_config
        self.users_to_test = list(graph.test_user_dict.keys())

    def __len__(self):
        return len(self.users_to_test)

    def __getitem__(self, index):
        batch_data = {}

        u_id = self.users_to_test[index]
        batch_data["u_id"] = u_id

        return batch_data
