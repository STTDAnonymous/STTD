from __future__ import division
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from utils import generate_dataset, get_normalized_adj, get_Laplace, calculate_random_walk_matrix,nb_zeroinflated_nll_loss,nb_zeroinflated_draw, nb_tweedie_nll_loss,nb_newtweedie_nll_loss,nb_zitweedie_nll_loss,nb_zitd_nll
from model import *
from utils import *
import random,os,copy
import math
import tqdm
from scipy.stats import nbinom
import pickle as pk
import os
# os.environ["CUDA_VISIBLE_DEVICES"]='2'
# Parameters
torch.manual_seed(0)
device = torch.device('cuda:3')
A = np.load('/home/ChenHao/MGTN/MGTN_jxk/OTHERS/STZINB-main/ny_data_full_15min/adj_rand0.npy') # change the loading folder
X = np.load('/home/ChenHao/MGTN/MGTN_jxk/OTHERS/STZINB-main/ny_data_full_15min/cta_samp_rand0.npy')

# A = np.load('/home/ChenHao/MGTN/MGTN_jxk/OTHERS/STZINB-main/ny_data_only10/adj_only10_rand0.npy') # change the loading folder
# X = np.load('/home/ChenHao/MGTN/MGTN_jxk/OTHERS/STZINB-main/ny_data_only10/cta_samp_only10_rand0.npy')
#
# A = np.load('/home/ChenHao/MGTN/MGTN_jxk/OTHERS/STZINB-main/cta_data_only10/adj_only10_rand0.npy') # change the loading folder
# X = np.load('/home/ChenHao/MGTN/MGTN_jxk/OTHERS/STZINB-main/cta_data_only10/cta_samp_only10_rand0.npy')

num_timesteps_output = 4
num_timesteps_input = 4

space_dim = X.shape[1]
batch_size = 10
hidden_dim_s = 20
hidden_dim_t = 20
rank_s = 20
rank_t = 4

epochs = 2000

# Initial networks
TCN1 = B_TCN(space_dim, hidden_dim_t, kernel_size=3, device=device).to(device=device)
TCN2 = B_TCN(hidden_dim_t, rank_t, kernel_size = 3, activation = 'linear', device=device).to(device=device)
TCN3 = B_TCN(rank_t, hidden_dim_t, kernel_size= 3, device=device).to(device=device)
TNB = NBNorm_ZeroInflated(hidden_dim_t,space_dim, four=True).to(device=device)
SCN1 = D_GCN(num_timesteps_input, hidden_dim_s, 3, att=False).to(device=device)
SCN2 = D_GCN(hidden_dim_s, rank_s, 3, activation = 'linear', att=True).to(device=device)
SCN3 = D_GCN(rank_s, hidden_dim_s, 2, att=True).to(device=device)
SNB = NBNorm_ZeroInflated(hidden_dim_s,num_timesteps_output, four=True).to(device=device)
STmodel = ST_new_TWEEDIE_ZeroInflated(SCN1, SCN2, SCN3, TCN1, TCN2, TCN3, SNB,TNB, four=True).to(device=device)

# Load data

X = X.T
X = X.astype(np.float32)
X = X.reshape((X.shape[0],1,X.shape[1]))

split_line1 = int(X.shape[2] * 0.60)
split_line2 = int(X.shape[2] * 0.70)
print(X.shape,A.shape)

# normalization
max_value = np.max(X[:, :, :split_line1])

train_original_data = X[:, :, :split_line1]
val_original_data = X[:, :, split_line1:split_line2]
test_original_data = X[:, :, split_line2:]
training_input, training_target = generate_dataset(train_original_data,
                                                    num_timesteps_input=num_timesteps_input,
                                                    num_timesteps_output=num_timesteps_output)
val_input, val_target = generate_dataset(val_original_data,
                                            num_timesteps_input=num_timesteps_input,
                                            num_timesteps_output=num_timesteps_output)
test_input, test_target = generate_dataset(test_original_data,
                                            num_timesteps_input=num_timesteps_input,
                                            num_timesteps_output=num_timesteps_output)
print('input shape: ',training_input.shape,val_input.shape,test_input.shape)


A_wave = get_normalized_adj(A)
A_q = torch.from_numpy((calculate_random_walk_matrix(A_wave).T).astype('float32'))
A_h = torch.from_numpy((calculate_random_walk_matrix(A_wave.T).T).astype('float32'))
A_q = A_q.to(device=device)
A_h = A_h.to(device=device)
# Define the training process
# criterion = nn.MSELoss()
optimizer = optim.Adam(STmodel.parameters(), lr=1e-4, weight_decay=1e-4)
training_nll   = []
validation_nll = []
validation_mae = []

for epoch in range(epochs):
    ## Step 1, training
    """
    # Begin training, similar training procedure from STGCN
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    """
    permutation = torch.randperm(training_input.shape[0])
    epoch_training_losses = []
    training_input = training_input.cuda()
    for i in range(0, training_input.shape[0], batch_size):
        STmodel.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        X_batch = X_batch.to(device=device)
        y_batch = y_batch.to(device=device)

        n_train,p_train,pi_train,zi_train = STmodel(X_batch,A_q,A_h)
        # loss = nb_zeroinflated_nll_loss(y_batch,n_train,p_train,pi_train)
        pi_train = torch.clip(pi_train, -15, 5)     # TODO 这个记得改回去！！fixme 一定
        loss = nb_zitd_nll(y_batch,n_train,p_train,pi_train,zi_train)
        # if torch.isnan(loss):
            # print(i)
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
        # print("epoch:{}, indices:{}, loss:{:.4f}".format(epoch, i, loss.item()))

    if epoch % 10 == 1:
        draw_3d_graph(y_batch.reshape(-1), phi=n_train.reshape(-1), rho=p_train.reshape(-1), mu=pi_train.reshape(-1))

    training_nll.append(sum(epoch_training_losses)/len(epoch_training_losses))
    training_input = training_input.cpu()
    torch.cuda.empty_cache()
    ## Step 2, validation
    with torch.no_grad():
        STmodel.eval()
        val_input = val_input.to(device=device)
        val_target = val_target.to(device=device)

        n_val,p_val,pi_val, zi_val = STmodel(val_input,A_q,A_h)
        val_loss = nb_zitd_nll(val_target,n_val,p_val,pi_val,zi_val).to(device="cpu")

        pi_val = torch.clip(pi_val, -10, 4)
        pi_val = torch.exp(pi_val)      # fixme

        print('Distribution_val,mean,min,max',torch.mean(pi_val),torch.min(pi_val),torch.max(pi_val))
        print('Pi_val,mean,min,max',torch.mean(pi_val),torch.min(pi_val),torch.max(pi_val))
        print('phi_val,mean,min,max',torch.mean(n_val),torch.min(n_val),torch.max(n_val))
        print('rou_val,mean,min,max',torch.mean(p_val),torch.min(p_val),torch.max(p_val))

        validation_nll.append(np.asscalar(val_loss.detach().numpy()))
        # Calculate the expectation value
        # val_pred = (1-pi_val.detach().cpu().numpy())*(n_val.detach().cpu().numpy()/p_val.detach().cpu().numpy()-n_val.detach().cpu().numpy()) # pipred
        # print(val_pred.mean(),pi_val.detach().cpu().numpy().min())
        # mae = np.mean(np.abs(val_pred - val_target.detach().cpu().numpy()))
        print_errors(val_target.detach().cpu().numpy(), (1-zi_val.detach().cpu().numpy()) * pi_val.detach().cpu().numpy())
        mae = np.mean(np.abs((1-zi_val.detach().cpu().numpy()) * pi_val.detach().cpu().numpy() - val_target.detach().cpu().numpy()))
        # mae = np.median(np.abs((1-zi_val.detach().cpu().numpy()) * pi_val.detach().cpu().numpy() - val_target.detach().cpu().numpy()))
        validation_mae.append(mae)
        # print(mae)
        n_val,p_val,pi_val = None,None,None
        val_input = val_input.to(device="cpu")
        val_target = val_target.to(device="cpu")
    print('Epoch: {}'.format(epoch))
    print("Training loss: {}".format(training_nll[-1]))
    print('Epoch %d: trainNLL %.5f; valNLL %.5f; mae %.4f'%(epoch,training_nll[-1],validation_nll[-1],validation_mae[-1]))
    if np.asscalar(training_nll[-1]) == min(training_nll):
        best_model = copy.deepcopy(STmodel.state_dict())
    checkpoint_path = "checkpoints/"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    with open("checkpoints/losses.pk", "wb") as fd:
        pk.dump((training_nll, validation_nll, validation_mae), fd)
    if np.isnan(training_nll[-1]):
        break

    if epoch % 1000 == 400:
        ##### Test #####
        print("TEST")
        STmodel.eval()
        with torch.no_grad():
            test_input = test_input.to(device='cpu')  # .to(device=device)
            test_target = test_target.to(device='cpu')  # .to(device=device)
            print(test_input.is_cuda, A_q.is_cuda, A_h.is_cuda)

            test_loss_all = []
            test_pred_all = np.zeros_like(test_target)
            n_test_all = np.zeros_like(test_target)
            p_test_all = np.zeros_like(test_target)
            pi_test_all = np.zeros_like(test_target)
            print(test_input.shape, test_target.shape)
            for i in range(0, test_input.shape[0], batch_size):
                x_batch = test_input[i:i + batch_size]
                x_batch = x_batch.to(device)
                n_test, p_test, pi_test, zi_test = STmodel(x_batch, A_q, A_h)
                pi_test = torch.clip(pi_test, -10, 4)
                test_loss = nb_zitd_nll(test_target[i:i + batch_size].to(device), n_test, p_test,
                                                   pi_test, zi_test).to(
                    device="cpu")
                test_loss = np.asscalar(test_loss.detach().numpy())

                pi_test = torch.exp(pi_test)
                mean_pred = (1-zi_test.detach().cpu().numpy()) * pi_test.detach().cpu().numpy()

                test_pred_all[i:i + batch_size] = mean_pred  # test_pred_all是均值
                n_test_all[i:i + batch_size] = n_test.detach().cpu().numpy()
                p_test_all[i:i + batch_size] = p_test.detach().cpu().numpy()
                pi_test_all[i:i + batch_size] = pi_test.detach().cpu().numpy()  # fixme
                test_loss_all.append(test_loss)

            # The error of each horizon
            mae_list = []
            rmse_list = []
            mape_list = []
            for horizon in range(test_pred_all.shape[2]):
                mae = np.mean(
                    np.abs(test_pred_all[:, :, horizon] - test_target[:, :, horizon].detach().cpu().numpy()))
                rmse = np.sqrt(
                    np.mean(test_pred_all[:, :, horizon] - test_target[:, :, horizon].detach().cpu().numpy()))
                mape = np.mean(
                    np.abs((test_pred_all[:, :, horizon] - test_target[:, :, horizon].detach().cpu().numpy()) / (
                            test_target[:, :, horizon].detach().cpu().numpy() + 1e-5)))
                # fixme 没有mae，rmse，mape这些情况
                mae_list.append(mae)
                rmse_list.append(rmse)
                mape_list.append(mape)
                print('Horizon %d MAE:%.4f RMSE:%.4f MAPE:%.4f' % (horizon, mae, rmse, mape))
            print('Overall score: NLL %.5f; mae %.4f; rmse %.4f; mape %.4f' % (
                test_loss, np.mean(mae_list), np.mean(rmse_list), np.mean(mape_list)))

np.savez_compressed('output/ny_full_5min_ZISTNB', target=test_target.detach().cpu().numpy(), max_value=max_value,
                    mean_pred=test_pred_all, n=n_test_all, p=p_test_all, pi=pi_test_all)

STmodel.load_state_dict(best_model)
torch.save(STmodel,'pth/STZINB_ny_full_5min.pth')
