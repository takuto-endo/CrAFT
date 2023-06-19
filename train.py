"""
train.py
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim

# original module
from utils import set_random_seed
from utils import count_parameters, classification_scores, mean_sq_error
from data_loader import DataSetCatCon
from models import CrAFT
from augmentations import embed_data_mask
# from pretrain import pretrain###


def tt_split(X, y, missing_mask, indices):
    x_d = {
        'data': X[indices],
        'mask': missing_mask[indices]
    }
    assert x_d['data'].shape == x_d['mask'].shape, 'Shape of data and mask are different'
    y_d = {
        'data': y[indices]
    }
    assert x_d['data'].shape[0] == y_d['data'].shape[0], 'Number of data and label are different'
    return x_d, y_d

def make_corr(X, lam, device):# X:numpy, lam:scaler
    X = torch.from_numpy(X)# X:tensor
    index = torch.randperm(len(X))
    corr = torch.from_numpy(np.random.choice(2,(X.shape),p=[lam,1-lam])).to(device)# 0:aug 1:noaug
    x = X[index,:]# 入れ替え元のデータ
    X_corr = X.clone().detach()
    X_corr[corr==0] = x[corr==0]
    return X_corr.numpy()

def train(X, y, missing_mask, train_indices, val_indices, test_indices, params, device):

    params['saved_model_path'] = []

    task = params['task']
    set_random_seed(params['set_seed'])
    if task == 'regression':
        y_dim = 1
    else:
        y_dim = len(np.unique(y))
    vision_dset = params['vision_dset']

    all_cm = np.array([[0,0],[0,0]])

    for fold in range(len(train_indices)):

        print('[Start] Fold: {}'.format(fold+1))
        model_name = f'{params["modelsave_path"]}/bestmodel_fold{fold+1}.pth'# '%s/bestmodel.pth' % (params["modelsave_path"])
        
        train_indice = train_indices[fold]
        val_indice = val_indices[fold]

        X_train, y_train = tt_split(X, y, missing_mask, train_indice)
        train_mean, train_std = np.array(X_train['data'][:,params['continuous_index']],dtype=np.float32).mean(0), np.array(X_train['data'][:,params['continuous_index']],dtype=np.float32).std(0)
        train_std = np.where(train_std < 1e-6, 1e-6, train_std)
        if train_mean is None:
            continuous_mean_std = None
        else:
            continuous_mean_std = np.array([train_mean,train_std]).astype(np.float32) 
        X_valid, y_valid = tt_split(X, y, missing_mask, val_indice)

        if params["corrupt_rate"] is not None:
            # cutmix 10~90%
            lam = params["corrupt_rate"]
            print(f'\nCutmix with {lam} rate.\n')
            X_train['data'] = make_corr(X_train['data'], lam, device)
            X_valid['data'] = make_corr(X_valid['data'], lam, device)

        print(f"X_train shape: {X_train['data'].shape}")
        print(f"y_train shape: {y_train['data'].shape}")
        print(f"X_valid shape: {X_valid['data'].shape}")
        print(f"y_valid shape: {y_valid['data'].shape}")

        train_ds = DataSetCatCon(X_train, y_train, params['categorical_index'],params['dtask'],continuous_mean_std)
        trainloader = DataLoader(train_ds, batch_size=params['batchsize'], shuffle=True,num_workers=0)#True

        valid_ds = DataSetCatCon(X_valid, y_valid, params['categorical_index'],params['dtask'],continuous_mean_std)
        validloader = DataLoader(valid_ds, batch_size=params['batchsize'], shuffle=False,num_workers=0)

        skip_test = params['skip_test_set']
        if not skip_test:
            X_test, y_test = tt_split(X, y, missing_mask, test_indices[fold])
            test_ds = DataSetCatCon(X_test, y_test, params['categorical_index'],params['dtask'],continuous_mean_std)
            testloader = DataLoader(test_ds, batch_size=params['batchsize'], shuffle=False,num_workers=0)
            print(f"X_test shape: {X_test['data'].shape}")
            print(f"y_test shape: {y_test['data'].shape}")

        # Train
        print(f'categories: {params["categorical_dims"]}')
        print(f'continuous: {params["continuous_index"]}')
        craft_params = {
            'categories': params['categorical_dims'],
            'num_continuous': len(params['continuous_index']),
            'dim': params['embedding_size'],
            'cont_embeddings': params['cont_embeddings'],
            'dim_out': 1,# check
            'y_dim': y_dim,

            'depth': params['transformer_depth'],
            'heads': params['attention_heads'],
            'attentiontype': params['attention_type'],
            'attn_dropout': params['attention_dropout'],
            'ff_dropout': params['ff_dropout'],
            'mlp_hidden_mults': (4,2),# check
            'final_mlp_style': params['final_mlp_style'],

            'cross_nhead': params['cross_nhead'],
            'cross_nhid': params['cross_nhid'],
            'cross_alpha': params['cross_alpha'],

            'random_seed': params['set_seed'],
            'ff_hidden': params['ff_hidden']
        }

        model = CrAFT(**craft_params).to(device)

        if y_dim == 2 and task == 'binary':
            criterion = nn.CrossEntropyLoss().to(device)
        elif y_dim > 2 and  task == 'multiclass':
            criterion = nn.CrossEntropyLoss().to(device)
        elif task == 'regression':
            criterion = nn.MSELoss().to(device)
        else:
            raise'case not written yet'

        if params['pretrain']:
            print('Start pretrain...')
            print(f'Pretrain epoch: {params["pretrain_epochs"]}')
            print("Skip")
            # model = pretrain(model, params['categorical_index'], X_train, y_train, continuous_mean_std, params, device)
            print('Pretrain finished.')

        # Optimizer
        if  params["optimizer"] == 'Adam':
            optimizer = optim.Adam(model.parameters(),lr=params["lr"])
        elif params["optimizer"] == 'AdamW':
            optimizer = optim.AdamW(model.parameters(),lr=params["lr"])

        # Train ############
        print('Start training...')
        print(f'Epoch: {params["epochs"]}')

        best_valid_accuracy = -np.inf
        best_valid_auroc = -np.inf
        best_valid_cm = np.array([[0,0],[0,0]])
        best_valid_rmse = np.inf

        best_test_accuracy = -np.inf
        best_test_auroc = -np.inf
        best_test_cm = np.array([[0,0],[0,0]])
        best_test_rmse = np.inf

        for epoch in range(params["epochs"]):
            model.train()
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                optimizer.zero_grad()
                # x_categ is the the categorical data, x_cont has continuous data, y_gts has ground truth ys. cat_mask is an array of ones same shape as x_categ and an additional column(corresponding to CLS token) set to 0s. con_mask is an array of ones same shape as x_cont. 
                x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)

                # We are converting the data to embeddings in the next step
                _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset)
                    
                reps = model.transformer(x_categ_enc, x_cont_enc)# Batch済の変数が渡る
                y_reps = reps[:,0,:]
                
                y_outs = model.mlpfory(y_reps)
                if task == 'regression':
                    loss = criterion(y_outs,y_gts) 
                else:
                    y_gts = y_gts.to(torch.long)
                    loss = criterion(y_outs,y_gts.squeeze()) 
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Validation
            model.eval()
            with torch.no_grad():
                ###
                if task in ['binary','multiclass']:
                    train_accuracy, train_auroc, train_std_batch, train_loss, train_cm = classification_scores(model, trainloader, device, task,vision_dset, criterion)
                    accuracy, auroc, std_batch, valid_loss, valid_cm = classification_scores(model, validloader, device, task,vision_dset, criterion)
                    if not skip_test:
                        test_accuracy, test_auroc, test_std_batch, test_loss, test_cm = classification_scores(model, testloader, device, task,vision_dset, criterion)

                    print('[EPOCH %d] TRAIN ACCURACY: %.3f, TRAIN AUROC: %.6f TRAIN STDACC: %.3f' %
                        (epoch + 1, train_accuracy,train_auroc, train_std_batch ))
                    print('[EPOCH %d] VALID ACCURACY: %.3f, VALID AUROC: %.6f VALID STDACC: %.3f' %
                        (epoch + 1, accuracy,auroc, std_batch ))
                    if not skip_test:
                        print('[EPOCH %d] TEST ACCURACY: %.3f, TEST AUROC: %.6f TEST STDACC: %.3f \n' %
                            (epoch + 1, test_accuracy,test_auroc, test_std_batch ))
                    
                    if task =='multiclass':
                        if accuracy > best_valid_accuracy:
                            best_valid_accuracy = accuracy
                            if not skip_test:
                                best_test_accuracy = test_accuracy
                            torch.save(model.state_dict(), model_name)
                    else:
                        #if accuracy > best_valid_accuracy:
                        #    best_valid_accuracy = accuracy
                        if auroc > best_valid_auroc:
                            best_valid_auroc = auroc
                            best_valid_accuracy = accuracy
                            best_valid_cm = valid_cm
                            if not skip_test:
                                best_test_auroc = test_auroc
                                best_test_accuracy = test_accuracy
                                best_test_cm = test_cm
                            torch.save(model.state_dict(), model_name)

                else:
                    train_rmse, train_std_batch, train_loss = mean_sq_error(model, trainloader, device,vision_dset, criterion)  
                    valid_rmse, valid_std_batch, valid_loss = mean_sq_error(model, validloader, device,vision_dset, criterion)    
                    if not skip_test:
                        test_rmse, test_std_batch, test_loss = mean_sq_error(model, testloader, device,vision_dset, criterion)  
                    print('[EPOCH %d] TRAIN RMSE: %.6f' %
                        (epoch + 1, train_rmse ))
                    print('[EPOCH %d] VALID RMSE: %.6f' %
                        (epoch + 1, valid_rmse ))
                    if not skip_test:
                        print('[EPOCH %d] TEST RMSE: %.6f' %
                            (epoch + 1, test_rmse ))
                    if valid_rmse < best_valid_rmse:
                        best_valid_rmse = valid_rmse
                        if not skip_test:
                            best_test_rmse = test_rmse
                        torch.save(model.state_dict(),model_name)
            model.train()

        print('Finished Training')

        params['saved_model_path'].append(model_name)
        total_parameters = count_parameters(model)
        print('TOTAL NUMBER OF PARAMS: %d' %(total_parameters))
        if task =='binary':
            if not skip_test:
                print('AUROC on best model:  %.6f' %(best_test_auroc))
                print('ACCC on best model:  %.6f' %(best_test_accuracy))
                # 混同行列を整形して表示する
                print("Confusion Matrix:")
                print("-----------------")
                print(f"{'':<8s}|{'Predicted Positive':<18s}|{'Predicted Negative':<18s}|")
                print("-----------------")
                print(f"{'Actual Positive':<8s}|{test_cm[0, 0]:<18d}|{test_cm[0, 1]:<18d}|")
                print(f"{'Actual Negative':<8s}|{test_cm[1, 0]:<18d}|{test_cm[1, 1]:<18d}|")
                all_cm[0, 0] += test_cm[0, 0]
                all_cm[0, 1] += test_cm[0, 1]
                all_cm[1, 0] += test_cm[1, 0]
                all_cm[1, 1] += test_cm[1, 1]
            else:
                print('AUROC on best model:  %.6f' %(best_valid_auroc))
                print('ACCC on best model:  %.6f' %(best_valid_accuracy))
                # 混同行列を整形して表示する
                print("Confusion Matrix:")
                print("-----------------")
                print(f"{'':<8s}|{'Predicted Positive':<18s}|{'Predicted Negative':<18s}|")
                print("-----------------")
                print(f"{'Actual Positive':<8s}|{best_valid_cm[0, 0]:<18d}|{best_valid_cm[0, 1]:<18d}|")
                print(f"{'Actual Negative':<8s}|{best_valid_cm[1, 0]:<18d}|{best_valid_cm[1, 1]:<18d}|")
                all_cm[0, 0] += best_valid_cm[0, 0]
                all_cm[0, 1] += best_valid_cm[0, 1]
                all_cm[1, 0] += best_valid_cm[1, 0]
                all_cm[1, 1] += best_valid_cm[1, 1]
        elif task =='multiclass':
            if not skip_test:
                print('Accuracy on best model:  %.3f' %(best_test_accuracy))
            else:
                print('Accuracy on best model:  %.3f' %(best_valid_accuracy))
        else:
            if not skip_test:
                print('RMSE on best model:  %.6f' %(best_test_rmse))
            else:
                print('RMSE on best model:  %.6f' %(best_valid_rmse))

    # finish all folds
    print('\nFinished all Training.')
    if task =='binary':
        # 混同行列を整形して表示する
        print("Confusion Matrix:")
        print("-----------------")
        print(f"{'':<8s}|{'Predicted Positive':<18s}|{'Predicted Negative':<18s}|")
        print("-----------------")
        print(f"{'Actual Positive':<8s}|{all_cm[0, 0]:<18d}|{all_cm[0, 1]:<18d}|")
        print(f"{'Actual Negative':<8s}|{all_cm[1, 0]:<18d}|{all_cm[1, 1]:<18d}|")

if __name__ == '__main__':
    # Do something
    pass