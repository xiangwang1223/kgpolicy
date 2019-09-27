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
from utility.test_model import test
from utility.helper import early_stopping, ensureDir, freeze, unfreeze
from utility.test_model import args_config, CKG

from dataloader.data_processor import CKG_Data
from dataloader.data_loader import build_loader

from recommender.MF import MF
from recommender.KGAT import KGAT
from sampler.KGPolicy_Sampler import KGPolicy


def train_dns_epoch(recommender, train_loader, recommender_optim, cur_epoch):
    loss, base_loss, reg_loss = 0, 0, 0

    tbar = tqdm(train_loader, ascii=True)
    for batch_data in tbar:
        tbar.set_description('Epoch {}'.format(cur_epoch))

        if torch.cuda.is_available():
            data_batch = {k: v.cuda(non_blocking=True) for k, v in batch_data.items()}

        recommender_optim.zero_grad()

        with torch.no_grad():
            users = batch_data["u_id"]
            negs = batch_data["neg_i_ids"]
            ranking = recommender.rank(users, negs)

        _, indices = torch.sort(ranking, descending=True)
        indices = indices[:, 0]

        batch_size = negs.size(0)

        row_id = torch.arange(batch_size, device=negs.device).unsqueeze(1)
        indices = indices.unsqueeze(1)
        good_neg = negs[row_id, indices].squeeze()

        batch_data["neg_i_id"] = good_neg
        _, loss_batch, gmf_batch_loss, reg_batch_loss = recommender(batch_data)

        loss_batch.backward()
        recommender_optim.step()

        loss += loss_batch
        base_loss += gmf_batch_loss
        reg_loss += reg_batch_loss

    print("Epoch {0:4d}: \n Training loss: [{1:4f} = {2:4f} + {3:4f}]\n".format(cur_epoch, loss, base_loss, reg_loss))

    return base_loss, base_loss, reg_loss

def train_random_epoch(recommender, train_loader, recommender_optim, cur_epoch):
    loss, base_loss, reg_loss = 0, 0, 0

    tbar = tqdm(train_loader, ascii=True)
    for batch_data in tbar:
        tbar.set_description('Epoch {}'.format(cur_epoch))

        if torch.cuda.is_available():
            data_batch = {k: v.cuda(non_blocking=True) for k, v in batch_data.items()}
        
        recommender_optim.zero_grad()

        _, loss_batch, gmf_batch_loss, reg_batch_loss = recommender(batch_data)

        loss_batch.backward()
        recommender_optim.step()

        loss += loss_batch
        base_loss += gmf_batch_loss
        reg_loss += reg_batch_loss

    print("Epoch {0:4d}: \n Training loss: [{1:4f} = {2:4f} + {3:4f}]\n".format(cur_epoch, loss, base_loss, reg_loss))

    return base_loss, base_loss, reg_loss

def train_one_epoch(recommender, sampler, 
                    train_loader, 
                    recommender_optim, sampler_optim, 
                    adj_matrix, edge_matrix, 
                    train_data, 
                    cur_epoch, 
                    avg_reward, 
                    config):

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
        selected_neg_items, _ = sampler(batch_data, adj_matrix, edge_matrix)
        users = batch_data["u_id"]
        neg = batch_data["neg_i_id"]
        pos = batch_data["pos_i_id"]

        """if output from sampler includes training items, replace it with goodneg"""
        train_set = train_data[users]
        in_train = torch.sum(selected_neg_items.unsqueeze(1)==train_set.long(), dim=1).byte()
        selected_neg_items[in_train] = neg[in_train]

        """Train recommender with sampled negative items"""
        if config.recommender == "KGAT":
            loss_batch, base_loss_batch, reg_loss_batch = recommender(users, pos, selected_neg_items, edge_matrix)
        elif config.recommender == "MF":
            loss_batch, base_loss_batch, reg_loss_batch = recommender(users, pos, selected_neg_items)
        else:
            raise Exception('Model has not been implemented')

        loss_batch.backward()
        recommender_optim.step()

        """Train sampler network"""
        sampler_optim.zero_grad()
        selected_neg_items, selected_neg_prob = sampler(batch_data, adj_matrix, edge_matrix)
        
        with torch.no_grad():
            reward_batch = recommender.get_reward(users, pos, selected_neg_items)

        epoch_reward += torch.sum(reward_batch)
        reward_batch -= avg_reward

        reinforce_loss = torch.sum(reward_batch * selected_neg_prob)
        reinforce_loss.backward()
        sampler_optim.step()
    
        """record loss in an epoch"""
        loss += loss_batch
        base_loss += base_loss_batch
        reg_loss += reg_loss_batch
    
    avg_reward = epoch_reward / num_batch
    print("Epoch {0:4d}: \n Training loss: [{1:4f} = {2:4f} + {3:4f}]\n Reward: {4:4f}".format(cur_epoch, loss, base_loss, reg_loss, avg_reward))
    
    return loss, base_loss, reg_loss, avg_reward

def build_sampler_graph(n_nodes, edge_threshold, graph):
    adj_matrix = torch.zeros(n_nodes, edge_threshold*2)
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
        sampled_edge += random.sample(range(CKG.item_range[0], CKG.item_range[1]+1), edge_threshold)
        adj_matrix[node] = torch.tensor(sampled_edge, dtype=torch.long)
        edge_matrix[node] = torch.tensor(edges, dtype=torch.long)

    if torch.cuda.is_available():
        adj_matrix = adj_matrix.cuda().long()
        edge_matrix = edge_matrix.cuda().long()

    return adj_matrix, edge_matrix


def train(train_loader, test_loader, data_config, args_config):

    """build training set"""
    train_mat = deepcopy(CKG.train_user_dict)

    num_user = max(train_mat.keys()) + 1
    num_true = max([len(i) for i in train_mat.values()])

    train_data = torch.zeros(num_user, num_true)

    """get padded tensor for training data"""
    for i in train_mat:
        true_list = train_mat[i]
        true_list += [-1] * (num_true - len(true_list))
        true_list = torch.tensor(true_list, dtype=torch.long)
        train_data[i] = true_list

    """preprocessing ckg graph"""
    params = {}

    ckg_file = "./output/ckg.pickle"
    graph = pickle.load(open(ckg_file, 'rb'))

    params["n_users"] = CKG.n_users
    params["n_relations"] = CKG.n_relations
    n_nodes = CKG.entity_range[1] + 1
    params["n_nodes"] = n_nodes
    params["item_range"] = CKG.item_range

    if args_config.pretrain_r:
        paras = torch.load(args_config.data_path + args_config.model_path)
        all_embed = torch.cat((paras["user_para"], paras["item_para"]))
        data_config["all_embed"] = all_embed
    
    data_config['n_nodes'] = n_nodes

    """Build Sampler and Recommender"""
    if args_config.recommender == "MF":
        recommender = MF(data_config=data_config, args_config=args_config)
    elif args_config.recommender == "KGAT":
        recommender = KGAT(data_config=data_config, args_config=args_config)
    
    if args_config.sampler == "KGPolicy":
        sampler = KGPolicy(recommender, params, args_config)
        sampler_optimer = torch.optim.SGD(sampler.parameters(), lr=args_config.slr)
    elif args_config.sampler == "DNS":
        pass    

    if torch.cuda.is_available():
        train_data = train_data.long().cuda()

        if args_config.sampler == "KGPolicy":
            sampler = sampler.cuda()
            print('\nSet sampler as: {}'.format(str(sampler)))
        recommender = recommender.cuda()
        print('Set recommender as: {}'.format(str(recommender)))

    """Build Optimizer"""
    recommender_optimer = torch.optim.Adam(recommender.parameters(), lr=args_config.rlr)

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
            adj_matrix, edge_matrix = build_sampler_graph(n_nodes, args_config.edge_threshold, graph)
        
        cur_epoch = epoch + 1
        t1 = time()
        if args_config.sampler == "KGPolicy":
            loss, base_loss, reg_loss, avg_reward = train_one_epoch(recommender, sampler, 
                                                                    train_loader, 
                                                                    recommender_optimer, sampler_optimer, 
                                                                    adj_matrix, edge_matrix, 
                                                                    train_data, 
                                                                    cur_epoch, 
                                                                    avg_reward, 
                                                                    args_config)
        elif args_config.sampler == "DNS": 
            loss, base_loss, reg_loss = train_dns_epoch(recommender, train_loader, recommender_optimer, cur_epoch)
        elif args_config.sampler == "Random":
            loss, base_loss, reg_loss = train_random_epoch(recommender, train_loader, recommender_optimer, cur_epoch)

        """Test"""
        if cur_epoch % args_config.show_step == 0:
            save_model('recommender.ckpt', recommender, args_config)
            with torch.no_grad():
                t2 = time()
                ret = test(recommender, test_loader)

            t3 = time()
            loss_loger.append(loss)
            rec_loger.append(ret['recall'])
            pre_loger.append(ret['precision'])
            ndcg_loger.append(ret['ndcg'])
            hit_loger.append(ret['hit_ratio'])

            perf_str = 'Evaluate[%.1fs]: \n    recall=[%.5f, %.5f, %.5f, %.5f, %.5f], ' \
                       '\n precision=[%.5f, %.5f, %.5f, %.5f, %.5f], \n       hit=[%.5f, %.5f, %.5f, %.5f, %.5f], \n      ndcg=[%.5f, %.5f, %.5f, %.5f, %.5f] \n' % \
                       (t3 - t2,
                        ret['recall'][0], ret['recall'][1], ret['recall'][2], ret['recall'][3], ret['recall'][4],
                        ret['precision'][0], ret['precision'][1], ret['precision'][2], ret['precision'][3], ret['precision'][4],
                        ret['hit_ratio'][0],ret['hit_ratio'][1], ret['hit_ratio'][2], ret['hit_ratio'][3], ret['hit_ratio'][4],
                        ret['ndcg'][0], ret['ndcg'][1], ret['ndcg'][2], ret['ndcg'][3], ret['ndcg'][4])
            print(perf_str)

            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=64)

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

def save_model(file_name, model, config):
    if not os.path.isdir(config.out_dir):
        os.mkdir(config.out_dir)
    
    model_file = Path(config.out_dir + file_name)
    model_file.touch(exist_ok=True)

    print("Saving model...")
    torch.save(model.state_dict(), model_file)


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
