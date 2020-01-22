import torch

import numpy as np
from tqdm import tqdm


def get_score(model, n_users, n_items, train_user_dict):
    u_e, i_e = torch.split(model.all_embed, [n_users, n_items])

    score_matrix = torch.matmul(u_e, i_e.t())
    for u, pos in train_user_dict.items():
        score_matrix[u][pos - n_users] = -1e5

    return score_matrix


def cal_ndcg(topk, test_set, num_pos, k):
    n = min(num_pos, k)
    nrange = np.arange(n) + 2
    idcg = np.sum(1 / np.log2(nrange))

    dcg = 0
    for i, s in enumerate(topk):
        if s in test_set:
            dcg += 1 / np.log2(i + 2)

    ndcg = dcg / idcg

    return ndcg


def test_v2(model, ks, ckg):
    ks = eval(ks)
    train_user_dict, test_user_dict = ckg.train_user_dict, ckg.test_user_dict

    n_users = ckg.n_users
    n_items = ckg.n_items
    n_test_users = len(test_user_dict)

    score_matrix = get_score(model, n_users, n_items, train_user_dict)

    n_k = len(ks)
    result = {
        "precision": np.zeros(n_k),
        "recall": np.zeros(n_k),
        "ndcg": np.zeros(n_k),
        "hit_ratio": np.zeros(n_k),
    }

    for i, k in enumerate(tqdm(ks, ascii=True, desc="Evaluate")):
        precision, recall, ndcg, hr = 0, 0, 0, 0
        _, topk_index = torch.topk(score_matrix, k)
        topk_index = topk_index.cpu().numpy() + n_users

        for test_u, gt_pos in test_user_dict.items():
            topk = topk_index[test_u]
            num_pos = len(gt_pos)

            topk_set = set(topk)
            test_set = set(gt_pos)
            num_hit = len(topk_set & test_set)

            precision += num_hit / k
            recall += num_hit / num_pos
            hr += 1 if num_hit > 0 else 0

            ndcg += cal_ndcg(topk, test_set, num_pos, k)

        result["precision"][i] = precision / n_test_users
        result["recall"][i] = recall / n_test_users
        result["ndcg"][i] = ndcg / n_test_users
        result["hit_ratio"][i] = hr / n_test_users

    return result
