"""
run.py
"""

import numpy as np
import argparse
import torch
import os

# original module
from data_loader import data_load, data_split
from train import train
from utils import set_random_seed

def main(params, dataset_id, task, seed=42):

    # set random seed
    set_random_seed(seed)
    # make model save directory
    params['modelsave_path'] = params['savemodelroot'] + f'/{dataset_id}/{task}/'
    os.makedirs(params['modelsave_path'], exist_ok=True)
    print(f'modelsave_path = {params["modelsave_path"]}')

    # load data: {done: [openml], todo: [libsvm, local]}
    X, y, missing_mask, categorical_index, continuous_index, categorical_dims = data_load(dataset_id, task, params, seed)
    categorical_dims = np.append(np.array([1]),np.array(categorical_dims)).astype(int) #Appending 1 for CLS token, this is later used to generate embeddings.
    params['categorical_dims'] = categorical_dims
    params['categorical_index'] = categorical_index
    params['continuous_index'] = continuous_index

    print(f'categorical index = {categorical_index}')
    print(f'continuous index = {continuous_index}')

    if task == 'regression':
        params['dtask'] = 'reg'
    else:
        params['dtask'] = 'clf'

    # data split >> X_list: k*[X_train, X_val, X_test] >> X_train: ["data": ~, "mask": ~]
    train_indices, val_indices, test_indices = data_split(X, y, missing_mask, params)

    # train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device = {device}')
    train(X, y, missing_mask, train_indices, val_indices, test_indices, params, device)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_id', default=42890, type=int)
    parser.add_argument('--task', default='binary', type=str, choices=['binary','multiclass','regression'])

    parser.add_argument('--data_type', default='openml', type=str, choices=['openml','local','libsvm'])
    parser.add_argument('--original_dataset_path', default=None, type=str)# ./data/train.csv
    parser.add_argument('--libsvm_name', default='frappe', type=str)# 0.1~0.9
    parser.add_argument('--corrupt_rate', default=None, type=float)# 0.1~0.9
    parser.add_argument('--vision_dset', action = 'store_true')

    parser.add_argument('--savemodelroot', default='./bestmodels', type=str)
    parser.add_argument('--run_name', default='testrun', type=str)
    parser.add_argument('--set_seed', default= 42 , type=int)
    parser.add_argument('--dset_seed', default= 5 , type=int)
    parser.add_argument('--active_log', action = 'store_true')

    parser.add_argument('--split_type', default='train_test', type=str, choices=['train_test','k_fold','time_series'])
    parser.add_argument('--k_fold', default=5, type=int)
    parser.add_argument('--k_fold_style', default='normal', type=str, choices=['normal','stratified','group','group_stratified'])
    parser.add_argument('--skip_test_set', action='store_true')

    
    # Model Arguments
    parser.add_argument('--cont_embeddings', default='MLP', type=str,choices = ['MLP','Noemb','pos_singleMLP'])
    parser.add_argument('--embedding_size', default=32, type=int)
    parser.add_argument('--transformer_depth', default=1, type=int)
    parser.add_argument('--attention_heads', default=8, type=int)
    parser.add_argument('--attention_dropout', default=0.1, type=float)
    parser.add_argument('--ff_dropout', default=0.1, type=float)
    parser.add_argument('--attention_type', default='colrow', type=str,choices = ['col','colrow','row','justmlp','attn','attnmlp'])
    parser.add_argument('--optimizer', default='AdamW', type=str,choices = ['AdamW','Adam','SGD'])
    parser.add_argument('--scheduler', default='cosine', type=str,choices = ['cosine','linear'])
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batchsize', default=256, type=int)
    parser.add_argument('--pretrain', default = True, action = 'store_true')
    parser.add_argument('--pretrain_epochs', default=50, type=int)
    parser.add_argument('--pt_tasks', default=['contrastive','denoising'], type=str,nargs='*',choices = ['contrastive','contrastive_sim','denoising'])
    parser.add_argument('--pt_aug', default=[], type=str,nargs='*',choices = ['mixup','cutmix'])
    parser.add_argument('--pt_aug_lam', default=0.1, type=float)
    parser.add_argument('--mixup_lam', default=0.3, type=float)
    parser.add_argument('--train_mask_prob', default=0, type=float)
    parser.add_argument('--mask_prob', default=0, type=float)
    parser.add_argument('--ssl_avail_y', default= 0, type=int)
    parser.add_argument('--pt_projhead_style', default='diff', type=str,choices = ['diff','same','nohead'])
    parser.add_argument('--nce_temp', default=0.7, type=float)
    parser.add_argument('--lam0', default=0.5, type=float)
    parser.add_argument('--lam1', default=10, type=float)
    parser.add_argument('--lam2', default=1, type=float)
    parser.add_argument('--lam3', default=10, type=float)
    parser.add_argument('--final_mlp_style', default='sep', type=str,choices = ['common','sep'])
    parser.add_argument('--cross_nhead', default= 4 , type=int)
    parser.add_argument('--cross_nhid', default= 2 , type=int)
    parser.add_argument('--cross_alpha', default= 1.5 , type=float)
    parser.add_argument('--train_data_limit', default=None, type=int)
    parser.add_argument('--ff_hidden', default=512, type=int)

    args = parser.parse_args()# opt >> args
    args_dict = vars(args)# args >> args_dict

    return args_dict



if __name__ == '__main__':
    
    # parse arguments
    args_dict = parse_arguments()
    print(args_dict)

    main(args_dict, args_dict['dataset_id'], args_dict['task'], seed=args_dict['set_seed'])

    