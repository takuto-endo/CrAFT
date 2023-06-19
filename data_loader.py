"""
dataloader.py

1. data_load_from_openml: load dataset from openml
2. data_load_from_local: load dataset from local << dataset_id=-1
>> data_load
"""
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

import openml
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold

# original module
from utils import set_random_seed

def original_data_load_from_openml(dataset_id):
    # load dataset
    dataset = openml.datasets.get_dataset(dataset_id)
    print(dataset)

    ########## Datasets that require individualized attention
    if dataset_id == 42890:
        dataset.default_target_attribute = 'Machine failure'
    assert dataset.default_target_attribute is not None, 'No target attribute'
    print(f'target name = {dataset.default_target_attribute}')

    # get data
    X, y, categorical_indicator, _ = dataset.get_data(
        dataset_format='dataframe', target=dataset.default_target_attribute)
    
    return X, y, categorical_indicator

def data_load_from_openml(dataset_id, task, seed=0):
    print('\n====== Loading dataset from openml... ======')

    # set random seed
    set_random_seed(seed)

    # get data
    X, y, categorical_indicator = original_data_load_from_openml(dataset_id)

    ########## If you need per-dataset adjustments, please implement them here
    drop_columns = []
    if dataset_id == 0:
        pass
    elif dataset_id == 42890:
        drop_columns = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        categorical_indicator = [True, False, False, False, False, False]

    if len(drop_columns) > 0:
        X = X.drop(columns=drop_columns)
    
    attribute_names = X.columns.values.tolist()
    assert len(categorical_indicator) == len(attribute_names), 'Length of categorical_indicator and attribute_names are different'
    categorical_columns = [attribute_names[i] for i, is_categorical in enumerate(categorical_indicator) if is_categorical]
    categorical_index = [i for i, is_categorical in enumerate(categorical_indicator) if is_categorical]
    continuous_columns = [attribute_names[i] for i, is_categorical in enumerate(categorical_indicator) if not is_categorical]
    continuous_index = [i for i, is_categorical in enumerate(categorical_indicator) if not is_categorical]

    # missing mask
    temp = X.fillna("MissingValue")
    missing_mask = temp.ne("MissingValue").astype(int).values

    # encode categorical columns
    categorical_dims = []
    if len(categorical_columns) > 0:
        X[categorical_columns] = X[categorical_columns].astype('object')
        for c in categorical_columns:
            X[c] = X[c].fillna('MissingValue')
            le = LabelEncoder()
            X[c] = le.fit_transform(X[c].values)
            categorical_dims.append(len(le.classes_))
    # fill missing values in continuous columns
    if len(continuous_columns) > 0:
        for c in continuous_columns:
            X[c].fillna(X[c].mean())

    # convert to numpy
    X = X.values
    y = y.values
    if task != 'regression':
        le = LabelEncoder()
        y = le.fit_transform(y)

    print('\n====== Dataset loaded. ======')
    print(f'X shape = {X.shape}')
    print(f'y shape = {y.shape}')
    print(f'missing mask shape = {missing_mask.shape}')
    print(f'attribute names = {attribute_names}')
    print(f'categorical index = {categorical_index}')
    print(f'continuous index = {continuous_index}')
    print('===============================\n')
    
    return X, y, missing_mask, categorical_index, continuous_index, categorical_dims

def data_load_from_local(dataset_id, task, params, seed=0):
    print('\n====== Loading dataset from local... ======')
    # porto-seguro

    # set random seed
    set_random_seed(seed)

    # get data
    df = pd.read_csv(params['original_dataset_path'])
    target = 'target'
    y = df['target']
    X = df.drop(columns=['id','target'],axis=1)

    categorical_index = []
    continuous_index = []
    categorical_dims = []

    for i, c in enumerate(X.columns):
        print(f'==  number{i} > {c}  ==  ')
        if '_cat' in c:
            # print(X[c].value_counts())
            nunique = X[c].nunique()
            print(f'   categorical: {c} >> {nunique}')

            l_enc = LabelEncoder() 
            X[c] = l_enc.fit_transform(X[c].values)
            
            categorical_dims.append(len(l_enc.classes_))
            categorical_index.append(i)

        elif '_bin' in c:
            nunique = X[c].nunique()
            print(f'   binary: {c} >> {nunique}')

            categorical_dims.append(nunique)
            categorical_index.append(i)

        else:
            print(f'   numeric: {c}')
            continuous_index.append(i)

    # missing mask
    temp = X.fillna("MissingValue")
    missing_mask = temp.ne("MissingValue").astype(int).values

    return X.values, y.values, missing_mask, categorical_index, continuous_index, categorical_dims
    
def data_load_from_libsvm(dataset_id, task, params, seed=0):
    print('\n====== Loading dataset from libsvm... ======')

    def decode_libsvm(line):
        columns = line.split(' ')
        map_func = lambda pair: (int(pair[0]), float(pair[1]))
        id, value = zip(*map(lambda col: map_func(col.split(':')), columns[1:]))
        sample = {'id': torch.LongTensor(id),
                  'value': torch.FloatTensor(value),
                  'y': float(columns[0])}
        return sample
    
    d_name = params['libsvm_name']
    train_path = f'../data/{d_name}/train.libsvm'
    valid_path = f'../data/{d_name}/valid.libsvm'
    test_path = f'../data/{d_name}/test.libsvm'

    # not implemented


def data_load(dataset_id, task, params, seed=0):
    if params['data_type']=='local':
        # load dataset from local
        X, y, missing_mask, categorical_index, continuous_index, categorical_dims = data_load_from_local(dataset_id, task, params, seed)
    elif params['data_type']=='openml':
        # load dataset from openml
        X, y, missing_mask, categorical_index, continuous_index, categorical_dims = data_load_from_openml(dataset_id, task, seed)
    elif params['data_type']=='libsvm':
        # load dataset from libsvm
        pass
        # X, y, missing_mask, categorical_index, continuous_index, categorical_dims = data_load_from_libsvm(dataset_id, task, params, seed)

    return X, y, missing_mask, categorical_index, continuous_index, categorical_dims

def train_test_split(X, y, missing_mask, params):

    TRA_VAL = params['skip_test_set']

    if TRA_VAL:
        datasplit=[.8, .2, .0]
        train_indices = []
        val_indices = []
        test_indices = None
    else:
        datasplit=[.65, .15, .2]
        train_indices = []
        val_indices = []
        test_indices = []


    assert sum(datasplit) == 1, 'Sum of datasplit is not 1'

    indices = np.arange(len(X))
    np.random.shuffle(indices)

    if TRA_VAL:
        train_indice = indices[:int(datasplit[0]*len(X))]
        val_indice = indices[int(datasplit[0]*len(X)):]
        train_indices.append(train_indice)
        val_indices.append(val_indice)
    else:
        train_indice = indices[:int(datasplit[0]*len(X))]
        val_indice = indices[int(datasplit[0]*len(X)):int((datasplit[0]+datasplit[1])*len(X))]
        test_indice = indices[int((datasplit[0]+datasplit[1])*len(X)):]
        train_indices.append(train_indice)
        val_indices.append(val_indice)
        test_indices.append(test_indice)

    return train_indices, val_indices, test_indices
    

def k_fold_split(X, y, missing_mask, params):
    
    TRA_VAL = params['skip_test_set']

    if TRA_VAL:
        train_indices = []
        val_indices = []
        test_indices = None
    else:
        test_rate = 0.2
        train_indices = []
        val_indices = []
        test_indices = []

    indices = np.arange(len(X))
    np.random.shuffle(indices)

    if TRA_VAL:
        data_num = len(X)
        fold_size = int(data_num/params['k_fold'])
    else:
        data_num = len(X) - int(len(X)*test_rate)# indices[:data_num]
        fold_size = int(data_num/params['k_fold'])
        test_indice = indices[data_num:]

    if params['k_fold_style'] == 'normal':
        kfold = KFold(n_splits=params['k_fold'])
        for train_indice, val_indice in kfold.split(indices[:data_num]):
            train_indices.append(indices[train_indice])
            val_indices.append(indices[val_indice])
            if not TRA_VAL:
                test_indices.append(test_indice)
    elif params['k_fold_style'] == 'stratified':
        kfold = StratifiedKFold(n_splits=params['k_fold'])
        for train_indice, val_indice in kfold.split(indices[:data_num], y[:data_num]):
            train_indices.append(indices[train_indice])
            val_indices.append(indices[val_indice])
            if not TRA_VAL:
                test_indices.append(test_indice)
        
    return train_indices, val_indices, test_indices


def data_split(X, y, missing_mask, params):
    # train-test split or k-fold split or time-series split
    if params['split_type'] == "train_test":
        train_indices, val_indices, test_indices = train_test_split(X, y, missing_mask, params)
    elif params['split_type'] == "k_fold":
        train_indices, val_indices, test_indices = k_fold_split(X, y, missing_mask, params)
    elif params['split_type'] == "time_series":
        pass

    return train_indices, val_indices, test_indices

class DataSetCatCon(Dataset):
    def __init__(self, X, Y, cat_cols,task='clf',continuous_mean_std=None):
        
        cat_cols = list(cat_cols)
        X_mask =  X['mask']
        X = X['data']

        con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))
        self.X1 = X[:,cat_cols].astype(np.int64) #categorical columns copy 負荷
        self.X2 = X[:,con_cols].astype(np.float32) #numerical columns
        self.X1_mask = X_mask[:,cat_cols].astype(np.int64) #categorical columns: maskの雛形生成
        self.X2_mask = X_mask[:,con_cols].astype(np.int64) #numerical columns: maskの雛形生成
        
        if task == 'clf':
            self.y = Y['data']#.astype(np.float32)
        else:
            self.y = Y['data'].astype(np.float32)
        self.cls = np.zeros((len(self.y), 1),dtype=int)# 出力と同じsize
        self.cls_mask = np.ones((len(self.y), 1),dtype=int)
        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std

    def __len__(self):# Dataset必須アイテム
        return len(self.y)
    
    def __getitem__(self, idx):# Dataset必須アイテム
        # X1 has categorical data, X2 has continuous
        return np.concatenate((self.cls[idx], self.X1[idx])), self.X2[idx],self.y[idx], np.concatenate((self.cls_mask[idx], self.X1_mask[idx])), self.X2_mask[idx]



if __name__ == '__main__':
    # Do something
    X, y, missing_mask, categorical_index, continuous_index, categorical_dims = data_load(dataset_id=42890, task='binart', seed=0)