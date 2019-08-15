#!/usr/local/bin/bash
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run KGPolicy2.")
    # ------------------------- experimental settings specific for data set --------------------------------------------
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Input weight path.')
    parser.add_argument('--data_path', nargs='?', default='../Data/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Input project path.')
    parser.add_argument('--dataset', nargs='?', default='amazon-book',
                        help='Choose a dataset.')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, 1: Pretrain with updating FISM variables, 2:Pretrain with fixed FISM variables.')
    parser.add_argument('--emb_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]',
                        help='Regularization for user and item embeddings.')
    parser.add_argument('--model_type', nargs='?', default='advnet',
                        help='Specify a loss type (pure_mf or gat_mf).')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')
    parser.add_argument('--layer_size', nargs='?', default='[64]',
                        help='Output sizes of every layer')

    # ------------------------- experimental settings specific for recommender --------------------------------------------
    parser.add_argument('--reward_type', nargs='?', default='pure',
                        help='reward function type: pure, prod')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--r_decay', type=float, default=0,
                        help='recommender weight decay.')

    # ------------------------- experimental settings specific for sampler --------------------------------------------
    parser.add_argument('--policy_type', nargs='?', default='uj',
                        help='policy function type: uj, uij')
    parser.add_argument('--s_decay', type=float, default=0,
                        help='Learning rate.')

    # ------------------------- experimental settings specific for training --------------------------------------------
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size for training.')
    parser.add_argument('--test_batch_size', type=int, default=2048,
                        help='batch size for test')
    parser.add_argument('--num_threads', type=int, default=4,
                        help='number of threads.')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epoch.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--show_step', type=int, default=10,
                        help='test step.')

    # ------------------------- experimental settings specific for testing ---------------------------------------------
    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]',
                        help='evaluate K list')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify a loss type (org, norm, or mean).')

    return parser.parse_args()