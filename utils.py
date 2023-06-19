import random
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil

# original module
from augmentations import embed_data_mask


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_default_mask(x):
    mask = np.ones_like(x)
    mask[:,-1] = 0
    return mask

def tag_gen(tag,y):
    return np.repeat(tag,len(y['data']))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  

def get_scheduler(args, optimizer):
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif args.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                      milestones=[args.epochs // 2.667, args.epochs // 1.6, args.epochs // 1.142], gamma=0.1)
    return scheduler

def imputations_acc_justy(model,dloader,device):
    model.eval()
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model)
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:,model.num_categories-1,:]
            y_outs = model.mlpfory(y_reps)
            # import ipdb; ipdb.set_trace()   
            y_test = torch.cat([y_test,x_categ[:,-1].float()],dim=0)
            y_pred = torch.cat([y_pred,torch.argmax(m(y_outs), dim=1).float()],dim=0)
            prob = torch.cat([prob,m(y_outs)[:,-1].float()],dim=0)
     
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]*100
    auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu())
    return acc, auc


def multiclass_acc_justy(model,dloader,device):
    model.eval()
    vision_dset = True
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:,model.num_categories-1,:]
            y_outs = model.mlpfory(y_reps)
            # import ipdb; ipdb.set_trace()   
            y_test = torch.cat([y_test,x_categ[:,-1].float()],dim=0)
            y_pred = torch.cat([y_pred,torch.argmax(m(y_outs), dim=1).float()],dim=0)
     
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]*100
    return acc, 0


def classification_scores(model, dloader, device, task,vision_dset, criterion,plot_attention=False, dset_id=None):
    model.eval()
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    batch_accs = []
    with torch.no_grad():
        running_loss = 0.0# new
        for i, data in enumerate(dloader, 0):
            #print(f'testloader {i} ------------------------------------------------------')
            x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)   

            if plot_attention:
                reps, col_att, row_att, cross_local_att, cross_grobal_att = model.transformer(x_categ_enc, x_cont_enc, plot_attention=plot_attention)
                if i==0:
                    col_atts = torch.sum(col_att, 0)
                    col_numbers = torch.ones(col_atts.size())*col_att.size()[0]

                    row_atts = torch.sum(row_att, 0)
                    row_numbers = torch.ones(row_atts.size())*row_att.size()[0]

                    cross_local_atts = torch.sum(cross_local_att, 0)
                    cross_local_numbers = torch.ones(cross_local_atts.size())*cross_local_att.size()[0]

                    cross_grobal_atts = cross_grobal_att.clone()
                    cross_grobal_numbers = torch.ones(cross_grobal_atts.size())
                else:
                    col_atts[:, :, :] += torch.sum(col_att, 0)
                    col_numbers[:, :, :] += col_att.size()[0]

                    row_atts[:, :row_att.size()[-1], :row_att.size()[-1]] += torch.sum(row_att, 0)
                    row_numbers[:, :row_att.size()[-1], :row_att.size()[-1]] += 1

                    cross_local_atts[:, :, :] += torch.sum(cross_local_att, 0)
                    cross_local_numbers[:, :, :] += cross_local_att.size()[0]

                    cross_grobal_atts[:, :, :] += cross_grobal_att
                    cross_grobal_numbers[:, :, :] += 1
            else:
                reps = model.transformer(x_categ_enc, x_cont_enc)

            y_reps = reps[:,0,:]# here!!!!!!!!!!!!
            # y_reps = torch.mean(reps, dim=1)
            y_outs = model.mlpfory(y_reps)

            y_gts = y_gts.to(torch.long)# new
            loss = criterion(y_outs, y_gts.squeeze())# new
            running_loss += loss.item()# new

            # ===
            correct_results_sum = (torch.argmax(y_outs, dim=1).float() == y_gts.squeeze().float()).sum().float()
            acc = correct_results_sum/y_gts.squeeze().float().shape[0]*100
            batch_accs.append(acc.cpu().numpy())
            # ===

            # import ipdb; ipdb.set_trace()   
            y_test = torch.cat([y_test,y_gts.squeeze().float()],dim=0)
            y_pred = torch.cat([y_pred,torch.argmax(y_outs, dim=1).float()],dim=0)
            if task == 'binary':
                prob = torch.cat([prob,m(y_outs)[:,-1].float()],dim=0)
        if plot_attention:
            
            if dset_id == None:
                dset_id = 'temp'

            save_fig_path = './figures/'+str(dset_id)
            if os.path.exists(save_fig_path):
                shutil.rmtree(save_fig_path)
                print(f'remove {save_fig_path}')
            os.makedirs(save_fig_path, exist_ok=True)

            os.makedirs(save_fig_path+'/col_att')
            col_atts = col_atts.to('cpu')/col_numbers.to('cpu')# attention_head * num_feature * num_featre
            for head_id in range(col_atts.size(0)):
                file_name = save_fig_path+"/col_att/head"+str(head_id)+".png"
                plt.figure(figsize=(18,18))
                sns.heatmap(col_atts[head_id,:,:])
                plt.savefig(file_name)
                plt.show()

                print('all col_atts:', torch.mean(col_atts[head_id,:,:],0))

            os.makedirs(save_fig_path+'/row_att')
            row_atts = row_atts.to('cpu')/row_numbers.to('cpu')# attention_head * batch_size * batch_size
            for head_id in range(row_atts.size(0)):
                file_name = save_fig_path+"/row_att/head"+str(head_id)+".png"
                plt.figure(figsize=(18,18))
                sns.heatmap(row_atts[head_id,:,:])
                plt.savefig(file_name)
                plt.show()

            os.makedirs(save_fig_path+'/cross_local_att')
            cross_local_atts = cross_local_atts.to('cpu')/cross_local_numbers.to('cpu')# cross_head * cross_nhid * num_feature
            for head_id in range(cross_local_atts.size(0)):
                file_name = save_fig_path+"/cross_local_att/head"+str(head_id)+".png"
                plt.figure(figsize=(18,6))
                sns.heatmap(cross_local_atts[head_id,:,:])
                plt.savefig(file_name)
                plt.show()

                print('all cross_local_atts:', torch.mean(cross_local_atts[head_id,:,:],0))

            os.makedirs(save_fig_path+'/cross_grobal_att')
            cross_grobal_atts = cross_grobal_atts.to('cpu')/cross_grobal_numbers.to('cpu')# cross_head * cross_nhid * num_feature
            for head_id in range(cross_grobal_atts.size(0)):
                file_name = save_fig_path+"/cross_grobal_att/head"+str(head_id)+".png"
                plt.figure(figsize=(18,6))
                sns.heatmap(cross_grobal_atts[head_id,:,:])
                plt.savefig(file_name)
                plt.show()

                print('all cross_grobal_atts:', torch.mean(cross_grobal_atts[head_id,:,:],0))

     
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]*100
    std_batch = np.std(np.array(batch_accs))
    auc = 0
    cm = np.array([[0,0],[0,0]])
    if task == 'binary':
        auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu())
        cm = confusion_matrix(y_pred=y_pred.cpu(), y_true=y_test.cpu())
    return acc.cpu().numpy(), auc, std_batch, running_loss, cm

def mean_sq_error(model, dloader, device, vision_dset, plot_attention=False):
    model.eval()
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    batch_rmses = []
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)

            if plot_attention:
                reps, col_att, row_att, cross_local_att, cross_grobal_att = model.transformer(x_categ_enc, x_cont_enc, plot_attention=plot_attention)
                if i==0:
                    col_atts = torch.sum(col_att, 0)
                    col_numbers = torch.ones(col_atts.size())*col_att.size()[0]

                    row_atts = torch.sum(row_att, 0)
                    row_numbers = torch.ones(row_atts.size())*row_att.size()[0]

                    cross_local_atts = torch.sum(cross_local_att, 0)
                    cross_local_numbers = torch.ones(cross_local_atts.size())*cross_local_att.size()[0]

                    cross_grobal_atts = cross_grobal_att.clone()
                    cross_grobal_numbers = torch.ones(cross_grobal_atts.size())
                else:
                    col_atts[:, :, :] += torch.sum(col_att, 0)
                    col_numbers[:, :, :] += col_att.size()[0]

                    row_atts[:, :row_att.size()[-1], :row_att.size()[-1]] += torch.sum(row_att, 0)
                    row_numbers[:, :row_att.size()[-1], :row_att.size()[-1]] += 1

                    cross_local_atts[:, :, :] += torch.sum(cross_local_att, 0)
                    cross_local_numbers[:, :, :] += cross_local_att.size()[0]

                    cross_grobal_atts[:, :, :] += cross_grobal_att
                    cross_grobal_numbers[:, :, :] += 1
            else:
                reps = model.transformer(x_categ_enc, x_cont_enc)
            # y_reps = reps[:,0,:]# here!!!!!!!!!!!!
            y_reps = torch.mean(reps, dim=1)
            y_outs = model.mlpfory(y_reps)

            # ===
            batch_rmse = mean_squared_error(y_gts.squeeze().float().cpu(), y_outs.cpu(), squared=False)
            batch_rmses.append(batch_rmse)
            # ===

            y_test = torch.cat([y_test,y_gts.squeeze().float()],dim=0)# y_groud truth
            y_pred = torch.cat([y_pred,y_outs],dim=0)

        if plot_attention:
            
            if dset_id == None:
                dset_id = 'temp'

            save_fig_path = './figures/'+str(dset_id)
            if os.path.exists(save_fig_path):
                shutil.rmtree(save_fig_path)
                print(f'remove {save_fig_path}')
            os.makedirs(save_fig_path, exist_ok=True)

            os.makedirs(save_fig_path+'/col_att')
            col_atts = col_atts.to('cpu')/col_numbers.to('cpu')# attention_head * num_feature * num_featre
            for head_id in range(col_atts.size(0)):
                file_name = save_fig_path+"/col_att/head"+str(head_id)+".png"
                plt.figure(figsize=(18,18))
                sns.heatmap(col_atts[head_id,:,:])
                plt.savefig(file_name)
                plt.show()

                print('all col_atts:', torch.mean(col_atts[head_id,:,:],0))

            os.makedirs(save_fig_path+'/row_att')
            row_atts = row_atts.to('cpu')/row_numbers.to('cpu')# attention_head * batch_size * batch_size
            for head_id in range(row_atts.size(0)):
                file_name = save_fig_path+"/row_att/head"+str(head_id)+".png"
                plt.figure(figsize=(18,18))
                sns.heatmap(row_atts[head_id,:,:])
                plt.savefig(file_name)
                plt.show()

            os.makedirs(save_fig_path+'/cross_local_att')
            cross_local_atts = cross_local_atts.to('cpu')/cross_local_numbers.to('cpu')# cross_head * cross_nhid * num_feature
            for head_id in range(cross_local_atts.size(0)):
                file_name = save_fig_path+"/cross_local_att/head"+str(head_id)+".png"
                plt.figure(figsize=(18,6))
                sns.heatmap(cross_local_atts[head_id,:,:])
                plt.savefig(file_name)
                plt.show()

                print('all cross_local_atts:', torch.mean(cross_local_atts[head_id,:,:],0))

            os.makedirs(save_fig_path+'/cross_grobal_att')
            cross_grobal_atts = cross_grobal_atts.to('cpu')/cross_grobal_numbers.to('cpu')# cross_head * cross_nhid * num_feature
            for head_id in range(cross_grobal_atts.size(0)):
                file_name = save_fig_path+"/cross_grobal_att/head"+str(head_id)+".png"
                plt.figure(figsize=(18,6))
                sns.heatmap(cross_grobal_atts[head_id,:,:])
                plt.savefig(file_name)
                plt.show()

                print('all cross_grobal_atts:', torch.mean(cross_grobal_atts[head_id,:,:],0))


        # import ipdb; ipdb.set_trace() 
        rmse = mean_squared_error(y_test.cpu(), y_pred.cpu(), squared=False)
        std_batch = np.std(np.array(batch_rmses))
        return rmse, std_batch

