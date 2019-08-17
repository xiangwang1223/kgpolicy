import os
import sys
from time import time
import numpy as np
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from copy import deepcopy

from utility.parser import parse_args
from utility.test_model import test
from utility.helper import early_stopping, ensureDir

from dataloader.data_processor import CKG_Data
from dataloader.loader_advnet import build_loader

from recommender.MF import MF
from sampler.KGPolicy_Sampler import KGPolicy

from utility.test_model import args_config, CKG


def train_one_epoch(recommender, sampler, train_loader, recommender_optim, sampler_optim, adj_matrix, train_data, cur_epoch, avg_reward):
    loss, base_loss, reg_loss = 0, 0, 0
    epoch_reward = 0

    """Train one epoch"""
    tbar = tqdm(train_loader, ascii=True)
    num_batch = len(train_loader)
    for _, batch_data in enumerate(tbar):
        tbar.set_description('Epoch {}'.format(cur_epoch))

        if torch.cuda.is_available():
            batch_data = {k: v.cuda(non_blocking=True) for k, v in batch_data.items()}

        """Train recommender using negtive item provided by sampler"""
        recommender_optim.zero_grad()
        selected_neg_items, selected_neg_prob = sampler(batch_data, adj_matrix)
        """filter items from trainset"""
        users = batch_data["u_id"]
        negs = batch_data["neg_i_id"]
        train_set = train_data[users]
        in_train = torch.sum(selected_neg_items.unsqueeze(1)==train_set.long(), dim=1).byte()

        selected_neg_items[in_train] = negs[in_train]

        # Train recommender with sampled negative items
        batch_data['neg_i_id'] = selected_neg_items

        _, loss_batch, base_loss_batch, reg_loss_batch = recommender(batch_data)
        loss_batch.backward()
        recommender_optim.step()

        """Train sampler network"""
        sampler_optim.zero_grad()
        with torch.no_grad():
            reward_batch, _, _, _ = recommender(batch_data)

        epoch_reward += torch.sum(reward_batch)
        reward_batch = reward_batch -avg_reward
        reinforce_loss = torch.sum(reward_batch * selected_neg_prob)
        reinforce_loss.backward()
        sampler_optim.step()

        """record loss in an epoch"""
        loss += loss_batch
        base_loss += base_loss_batch
        reg_loss += reg_loss_batch
    
    avg_reward = epoch_reward / num_batch
    
    print("Epoch {0:4d}: \n Training loss: [{1:4f} = {2:4f} + {3:4f}]\n".format(cur_epoch, loss, base_loss, reg_loss))
    
    return loss, base_loss, reg_loss, avg_reward

def build_adj(n_nodes, edge_threshold, graph):
    """build adjacency matrix, using random items as padding"""
    adj_matrix = torch.zeros(n_nodes, edge_threshold)

    for node in tqdm(graph.nodes, ascii=True, desc="Building adj matrix"):
        neighbors = list(graph.neighbors(node))
        padding_size = edge_threshold - len(neighbors)
        for _ in range(padding_size):
            neg_id = random.randint(CKG.item_range[0], CKG.item_range[1])
            neighbors.append(neg_id)
        neighbors = torch.tensor(neighbors, dtype=torch.long)
        adj_matrix[node] = neighbors

    if torch.cuda.is_available():
        adj_matrix = adj_matrix.cuda().long()

    return adj_matrix


def train(train_loader, test_loader, data_config, args_config):
    """build training set"""
    train_mat = deepcopy(CKG.train_user_dict)

    num_user = max(train_mat.keys()) + 1
    num_true = max([len(i) for i in train_mat.values()])

    train_data = torch.zeros(num_user, num_true)

    for i in train_mat:
        true_list = train_mat[i]
        true_list += [0] * (num_true - len(true_list))
        true_list = torch.tensor(true_list, dtype=torch.long)
        train_data[i] = true_list

    """preprocessing ckg graph"""
    params = {}
    graph = deepcopy(CKG.ckg_graph)

    """remove node with more than edge_threshold neighbor"""
    general_node = []
    for node in graph.nodes:
        if(len(set(graph.neighbors(node)))) > args_config.edge_threshold:
            general_node.append(node)
    graph.remove_nodes_from(general_node)

    edges = torch.tensor(list(graph.edges), dtype=torch.long)
    edges = edges[:, :2]
    if torch.cuda.is_available():
        edges = edges.cuda()
        train_data = train_data.long().cuda()

    params["edges"] = edges
    params["n_users"] = CKG.n_users
    n_nodes = CKG.entity_range[1] + 1
    params["n_nodes"] = n_nodes
    params["item_range"] = CKG.item_range

    """Build Sampler and Recommender"""
    recommender = MF(data_config=data_config, args_config=args_config)
    sampler = KGPolicy(recommender, params, args_config)
    if torch.cuda.is_available():
        sampler = sampler.cuda()
        recommender = recommender.cuda()

    print('\nSet sampler as: {}'.format(str(sampler)))
    print('Set recommender as: {}'.format(str(recommender)))

    """Build Optimizer"""
    sampler_optimer = torch.optim.Adam(sampler.parameters(), lr=args_config.lr, weight_decay=args_config.s_decay)
    recommender_optimer = torch.optim.Adam(recommender.parameters(), lr=args_config.lr, weight_decay=args_config.r_decay)

    """Initialize Best Hit Rate"""
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False
    cur_best_pre_0 = 0.
    t0 = time()
    avg_reward = 0

    for epoch in range(args_config.epoch):
        """build adjacency matrix"""
        if epoch % args_config.adj_epoch == 0:
            adj_matrix = build_adj(n_nodes, args_config.edge_threshold, graph)

        cur_epoch = epoch + 1
        t1 = time()
        loss, base_loss, reg_loss, avg_reward = train_one_epoch(recommender, sampler, train_loader, recommender_optimer, sampler_optimer, adj_matrix, train_data, cur_epoch, avg_reward)

        """Test"""
        if cur_epoch % args_config.show_step == 0:
            with torch.no_grad():
                t2 = time()
                ret = test(recommender, test_loader)

            t3 = time()
            loss_loger.append(loss)
            rec_loger.append(ret['recall'])
            pre_loger.append(ret['precision'])
            ndcg_loger.append(ret['ndcg'])
            hit_loger.append(ret['hit_ratio'])

            if args_config.verbose > 0:
                perf_str = 'Evaluate[%.1fs]: \n recall=[%.5f, %.5f], ' \
                           '\n precision=[%.5f, %.5f], \n hit=[%.5f, %.5f], \n ndcg=[%.5f, %.5f] \n' % \
                           (t3 - t2,
                            ret['recall'][0], ret['recall'][-1],
                            ret['precision'][0], ret['precision'][-1],
                            ret['hit_ratio'][0], ret['hit_ratio'][-1],
                            ret['ndcg'][0], ret['ndcg'][-1])
                print(perf_str)

            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=5)

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            if should_stop == True:
                break

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\n recall=[%s] \n precision=[%s] \n hit=[%s] \n ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)

if __name__ == '__main__':
    # fix the random seed.
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # set the gpu id.
    if torch.cuda.is_available():
        torch.cuda.set_device(args_config.gpu_id)

    # initialize the data config.
    data_config = {'n_users': CKG.n_users,'n_items': CKG.n_items,
                   'n_relations': CKG.n_relations + 2, 'n_entities': CKG.n_entities, }

    train_loader, test_loader = build_loader(args_config=args_config)
    train(train_loader=train_loader, test_loader=test_loader,
          data_config=data_config, args_config=args_config)
