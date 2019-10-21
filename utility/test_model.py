import torch

from tqdm import tqdm
import heapq
import multiprocessing
import numpy as np

from dataloader.data_loader import build_loader
import utility.metrics as metrics


def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i - _n_users]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    return r


def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i - _n_users]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    return r


def get_performance(user_pos_test, r, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio)}


def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    # uid
    u = x[1]

    # user u's items in the training set
    try:
        training_items = _train_user_dict[u.item()]
    except Exception:
        training_items = []
    # user u's items in the test set
    user_pos_test = _test_user_dict[u.item()]
    all_items = set(range(_item_range[0], _item_range[1] + 1))
    test_items = list(all_items - set(training_items))
    r = ranklist_by_heapq(user_pos_test, test_items, rating, _Ks)

    return get_performance(user_pos_test, r, _Ks)


def test(model, test_loader, ks, ckg):
    global _Ks, _train_user_dict, _test_user_dict, _n_users, _n_test_users, _item_range
    _Ks = eval(ks)
    _train_user_dict, _test_user_dict = ckg.train_user_dict, ckg.test_user_dict
    _n_users = ckg.n_users
    _n_test_users = len(_test_user_dict.keys())
    _item_range = ckg.item_range

    result = {'precision': np.zeros(len(_Ks)), 'recall': np.zeros(len(_Ks)), 'ndcg': np.zeros(len(_Ks)),
              'hit_ratio': np.zeros(len(_Ks))}

    cores = multiprocessing.cpu_count() // 2
    pool = multiprocessing.Pool(cores)

    for _, batch_data in enumerate(tqdm(test_loader, ascii=True, desc='Evaluate')):
        batch_u_id = batch_data['u_id']

        batch_pred = model.inference(batch_u_id)

        batch_pred = batch_pred.cpu().numpy()
        batch_u_id = batch_u_id.cpu().numpy()

        user_batch_rating_uid = zip(batch_pred, batch_u_id)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)

        for re in batch_result:
            result['precision'] += re['precision']/_n_test_users
            result['recall'] += re['recall']/_n_test_users
            result['ndcg'] += re['ndcg']/_n_test_users
            result['hit_ratio'] += re['hit_ratio']/_n_test_users

    pool.close()
    return result
