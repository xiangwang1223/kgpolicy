import os
import sys
import random
from time import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from copy import deepcopy
import pickle

from utility.parser import parse_args
from utility.test_model import test_v2
from utility.helper import early_stopping
from utility.parser import parse_args

from dataloader.data_loader import build_loader
from dataloader.data_processor import CKGData

from recommender.MF import MF
from sampler.KGPolicy_Sampler import KGPolicy


def train_one_epoch(recommender, sampler, 
                    train_loader, 
                    recommender_optim, sampler_optim, 
                    adj_matrix, edge_matrix, 
                    train_data, 
                    cur_epoch, 
                    avg_reward):

    loss, base_loss, reg_loss = 0, 0, 0
    epoch_reward = 0

    """Train one epoch"""
    tbar = tqdm(train_loader, ascii=True)
    num_batch = len(train_loader)
    for batch_data in tbar:
        
        tbar.set_description('Epoch {}'.format(cur_epoch))

        if torch.cuda.is_available():
            batch_data = {k: v.cuda(non_blocking=True) for k, v in batch_data.items()}

        """Train recommender using negtive item provided by sampler"""
        recommender_optim.zero_grad()

        users = batch_data["u_id"]
        neg = batch_data["neg_i_id"]
        pos = batch_data["pos_i_id"]

        selected_neg_items_list, _ = sampler(batch_data, adj_matrix, edge_matrix)
        selected_neg_items = selected_neg_items_list[-1, :]

        train_set = train_data[users]
        in_train = torch.sum(selected_neg_items.unsqueeze(1) == train_set.long(), dim=1).byte()
        selected_neg_items[in_train] = neg[in_train]

        base_loss_batch, reg_loss_batch = recommender(users, pos, selected_neg_items)
        loss_batch = base_loss_batch + reg_loss_batch

        loss_batch.backward()
        recommender_optim.step()

        """Train sampler network"""
        sampler_optim.zero_grad()
        selected_neg_items_list, selected_neg_prob_list = sampler(batch_data, adj_matrix, edge_matrix)
        
        with torch.no_grad():
            reward_batch = recommender.get_reward(users, pos, selected_neg_items_list)

        epoch_reward += torch.sum(reward_batch)
        reward_batch -= avg_reward

        batch_size = reward_batch.size(1)
        n = reward_batch.size(0) - 1
        R = torch.zeros(batch_size, device=reward_batch.device)
        reward = torch.zeros(reward_batch.size(), device=reward_batch.device)

        gamma = args_config.gamma

        for i, r in enumerate(reward_batch.flip(0)):
            R = r + gamma * R
            reward[n - i] = R

        reinforce_loss = -1 * torch.sum(reward_batch * selected_neg_prob_list)
        reinforce_loss.backward()
        sampler_optim.step()
    
        """record loss in an epoch"""
        loss += loss_batch
        base_loss += base_loss_batch
        reg_loss += reg_loss_batch
    
    avg_reward = epoch_reward / num_batch
    print(' Epoch {0:4d}: \
            \n  Training loss: [{1:4f} = {2:4f} + {3:4f}] \
            \n         Reward: {4:4f}'.format(cur_epoch, loss, base_loss, reg_loss, avg_reward))
    
    return loss, base_loss, reg_loss, avg_reward


def save_model(file_name, model, config):
    if not os.path.isdir(config.out_dir):
        os.mkdir(config.out_dir)
    
    model_file = Path(config.out_dir + file_name)
    model_file.touch(exist_ok=True)

    print("Saving model...")
    torch.save(model.state_dict(), model_file)


def build_sampler_graph(n_nodes, edge_threshold, graph):
    # adj_matrix = torch.zeros(n_nodes, edge_threshold*2)
    adj_matrix = torch.zeros(n_nodes, edge_threshold)
    edge_matrix = torch.zeros(n_nodes, edge_threshold)

    """sample neighbors for each node"""
    for node in tqdm(graph.nodes, ascii=True, desc="Build sampler matrix"):
        neighbors = list(graph.neighbors(node))
        if len(neighbors) >= edge_threshold:
            sampled_edge = random.sample(neighbors, edge_threshold)
            edges = deepcopy(sampled_edge)
        else:
            neg_id = random.sample(range(CKG.item_range[0], CKG.item_range[1]+1), edge_threshold-len(neighbors))
            node_id = [node]*(edge_threshold-len(neighbors))
            sampled_edge = neighbors + neg_id
            edges = neighbors + node_id
        
        """concatenate sampled edge with random edge"""
        # sampled_edge += random.sample(range(CKG.item_range[0], CKG.item_range[1]+1), edge_threshold)
        
        adj_matrix[node] = torch.tensor(sampled_edge, dtype=torch.long)
        edge_matrix[node] = torch.tensor(edges, dtype=torch.long)

    if torch.cuda.is_available():
        adj_matrix = adj_matrix.cuda().long()
        edge_matrix = edge_matrix.cuda().long()

    return adj_matrix, edge_matrix


def build_train_data(train_mat):
    num_user = max(train_mat.keys()) + 1
    num_true = max([len(i) for i in train_mat.values()])

    train_data = torch.zeros(num_user, num_true)

    for i in train_mat.keys():
        true_list = train_mat[i]
        true_list += [-1] * (num_true - len(true_list))
        train_data[i] = torch.tensor(true_list, dtype=torch.long)

    return train_data


def train(train_loader, test_loader, graph, data_config, args_config):
    """build padded training set"""
    train_mat = graph.train_user_dict
    train_data = build_train_data(train_mat)

    if args_config.pretrain_r:
        print("\nLoad model from {}".format(args_config.data_path + args_config.model_path))
        paras = torch.load(args_config.data_path + args_config.model_path)
        all_embed = torch.cat((paras["user_para"], paras["item_para"]))
        data_config["all_embed"] = all_embed
    
    recommender = MF(data_config=data_config, args_config=args_config)
    sampler = KGPolicy(recommender, data_config, args_config)

    if torch.cuda.is_available():
        train_data = train_data.long().cuda()
        sampler = sampler.cuda()
        recommender = recommender.cuda()

        print('\nSet sampler as: {}'.format(str(sampler)))
        print('Set recommender as: {}\n'.format(str(recommender)))

    recommender_optimer = torch.optim.Adam(recommender.parameters(), lr=args_config.rlr)
    sampler_optimer = torch.optim.Adam(sampler.parameters(), lr=args_config.slr)

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step, cur_best_pre_0, avg_reward = 0, 0., 0
    t0 = time()

    for epoch in range(args_config.epoch):
        if epoch % args_config.adj_epoch == 0:
            """sample adjacency matrix"""
            adj_matrix, edge_matrix = build_sampler_graph(data_config['n_nodes'],
                                                          args_config.edge_threshold, graph.ckg_graph)
        
        cur_epoch = epoch + 1
        loss, base_loss, reg_loss, avg_reward = train_one_epoch(recommender, sampler,
                                                                train_loader, 
                                                                recommender_optimer, sampler_optimer, 
                                                                adj_matrix, edge_matrix, 
                                                                train_data, 
                                                                cur_epoch, 
                                                                avg_reward)


        """Test"""
        if cur_epoch % args_config.show_step == 0:
            with torch.no_grad():
                t2 = time()
                ret = test_v2(recommender, args_config.Ks, graph)

            t3 = time()
            # loss_loger.append(loss)
            rec_loger.append(ret['recall'])
            pre_loger.append(ret['precision'])
            ndcg_loger.append(ret['ndcg'])
            hit_loger.append(ret['hit_ratio'])

            perf_str = 'Evaluate[%.1fs]: \
                        \n      recall=[%.5f, %.5f, %.5f, %.5f, %.5f], \
                        \n   precision=[%.5f, %.5f, %.5f, %.5f, %.5f], \
                        \n         hit=[%.5f, %.5f, %.5f, %.5f, %.5f], \
                        \n        ndcg=[%.5f, %.5f, %.5f, %.5f, %.5f] ' % \
                       (t3 - t2,
                        ret['recall'][0], ret['recall'][1],
                        ret['recall'][2], ret['recall'][3], ret['recall'][4],
                        ret['precision'][0], ret['precision'][1],
                        ret['precision'][2], ret['precision'][3], ret['precision'][4],
                        ret['hit_ratio'][0], ret['hit_ratio'][1],
                        ret['hit_ratio'][2], ret['hit_ratio'][3], ret['hit_ratio'][4],
                        ret['ndcg'][0], ret['ndcg'][1],
                        ret['ndcg'][2], ret['ndcg'][3], ret['ndcg'][4])
            print(perf_str)

            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=args_config.flag_step)

            if should_stop:
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
    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
   
    """initialize args and dataset"""
    args_config = parse_args()
    CKG = CKGData(args_config)
    
    """set the gpu id"""
    if torch.cuda.is_available():
        torch.cuda.set_device(args_config.gpu_id)
    
    data_config = {'n_users': CKG.n_users,
                   'n_items': CKG.n_items,
                   'n_relations': CKG.n_relations + 2, 
                   'n_entities': CKG.n_entities,
                   'n_nodes': CKG.entity_range[1] + 1,
                   'item_range': CKG.item_range}

    print('\ncopying CKG graph for data_loader.. it might take a few minutes')
    graph = deepcopy(CKG)
    train_loader, test_loader = build_loader(args_config=args_config, graph=graph)
    
    train(train_loader=train_loader, 
          test_loader=test_loader,
          graph=CKG,
          data_config=data_config, 
          args_config=args_config)
