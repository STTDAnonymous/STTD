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
from utils import generate_dataset, get_normalized_adj, get_Laplace, calculate_random_walk_matrix,nb_zeroinflated_nll_loss,nb_zeroinflated_draw, nb_tweedie_nll_loss
from model import *
from utils import *
import heapq
import random,os,copy
import math
import tqdm
from scipy.stats import nbinom
import pickle as pk
import os
# os.environ["CUDA_VISIBLE_DEVICES"]='2'
# Parameters
torch.manual_seed(0)
device = torch.device('cuda:0')
# 先导入邻接矩阵
A = np.load('STTD-main/ny_data_full_15min/adj_rand0.npy') # change the loading folder
# 再导入特矩阵
X = np.load('STTD-main/ny_data_full_15min/cta_samp_rand0.npy')

num_timesteps_output = 3                            # 预测输出天数
num_timesteps_input = num_timesteps_output          # 预测输入天数

space_dim = X.shape[1]
batch_size = 40                                       # 每次输入的batch_size
hidden_dim_s = 42                                    # spatial的hidden
hidden_dim_t = 42                                    # temporal的hidden
rank_s = 20                                         # todo
rank_t = 4                                          # todo

epochs = 1

# Initial networks
# TCN是对每一个节点的temporal编码
TCN1 = B_TCN(space_dim, hidden_dim_t, kernel_size=3, device=device).to(device=device)
TCN2 = B_TCN(hidden_dim_t, rank_t, kernel_size = 3, activation = 'linear', device=device).to(device=device)
TCN3 = B_TCN(rank_t, hidden_dim_t, kernel_size= 3, device=device).to(device=device)
# TNB是把模型输出修改为固定模式
TNB = NBNorm_ZeroInflated(hidden_dim_t,space_dim).to(device=device)
# SCN是spatial的编码
SCN1 = D_GCN(num_timesteps_input, hidden_dim_s, 3).to(device=device)
SCN2 = D_GCN(hidden_dim_s, rank_s, 2, activation = 'linear').to(device=device)
SCN3 = D_GCN(rank_s, hidden_dim_s, 2).to(device=device)
# SNB把模型输出修改为固定模式
SNB = NBNorm_ZeroInflated(hidden_dim_s,num_timesteps_output).to(device=device)
# 最后参数推导
STmodel = ST_NB_ZeroInflated(SCN1, SCN2, SCN3, TCN1, TCN2, TCN3, SNB,TNB).to(device=device)

# Load dataset

X = X.T
X = X.astype(np.float32)
X = X.reshape((X.shape[0],1,X.shape[1]))

# 划分train-test-valid集
split_line1 = int(X.shape[2] * 0.6)
split_line2 = int(X.shape[2] * 0.7)
print(X.shape,A.shape)

# normalization
max_value = np.max(X[:, :, :split_line1])           # 得到训练集的最大值

train_original_data = X[:, :, :split_line1]
val_original_data = X[:, :, split_line1:split_line2]
test_original_data = X[:, :, split_line2:]
# generate_dataset的意思是把模型固定为一个格式
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

# 归一化adj
A_wave = get_normalized_adj(A)
A_q = torch.from_numpy((calculate_random_walk_matrix(A_wave).T).astype('float32'))
A_h = torch.from_numpy((calculate_random_walk_matrix(A_wave.T).T).astype('float32'))
A_q = A_q.to(device=device)
A_h = A_h.to(device=device)
# Define the training process
# criterion = nn.MSELoss()
optimizer = optim.Adam(STmodel.parameters(), lr=1e-4)       # 创建优化器
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
    permutation = torch.randperm(training_input.shape[0])           # 全排列
    epoch_training_losses = []
    for i in range(0, training_input.shape[0], batch_size):
        STmodel.train()                                             # 模型训练 TODO
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        X_batch = X_batch.to(device=device)
        y_batch = y_batch.to(device=device)

        n_train,p_train,pi_train = STmodel(X_batch.cuda(),A_q,A_h)  # 得到输出
        loss = nb_zeroinflated_nll_loss(y_batch,n_train,p_train,pi_train)   # 计算损失
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
        # print("epoch:{}, indices:{}, loss:{:.4f}".format(epoch, i, loss.item()))
    training_nll.append(sum(epoch_training_losses)/len(epoch_training_losses))
    torch.cuda.empty_cache()
    ## Step 2, validation
    with torch.no_grad():
        STmodel.eval()
        val_input = val_input.to(device=device)
        val_target = val_target.to(device=device)

        n_val,p_val,pi_val = STmodel(val_input,A_q,A_h)
        print('Pi_val,mean,min,max',torch.mean(pi_val),torch.min(pi_val),torch.max(pi_val))
        val_loss = nb_zeroinflated_nll_loss(val_target,n_val,p_val,pi_val).to(device="cpu")
        validation_nll.append(np.asscalar(val_loss.detach().numpy()))

        # Calculate the expectation value        
        val_pred = (1-pi_val.detach().cpu().numpy())*(n_val.detach().cpu().numpy()/p_val.detach().cpu().numpy()-n_val.detach().cpu().numpy()) # pipred
        # print(val_pred.mean(),pi_val.detach().cpu().numpy().min())
        mae = np.mean(np.abs(val_pred - val_target.detach().cpu().numpy()))

        print_errors(val_target.detach().cpu().numpy(), val_pred)

        # mae = np.median(np.abs(val_pred - val_target.detach().cpu().numpy()))
        print(mae)
        validation_mae.append(mae)

        n_val,p_val,pi_val = None,None,None
        val_input = val_input.to(device="cpu")
        val_target = val_target.to(device="cpu")

        hit_rate = 0.2
        val_target1 = val_target.reshape(-1)
        non_zero_mask = val_target1.detach().cpu().numpy()>0
        pred1 = val_pred
        pred1 = pred1.reshape(-1)
        # pred_index = heapq.nlargest(20, range(len(pred1)), pred1.__getitem__)
        pred_index = heapq.nlargest(int(hit_rate*len(pred1)), range(len(pred1)), pred1.__getitem__)

        hit_mask = np.zeros_like(non_zero_mask)
        hit_mask[pred_index] = 1

        ans = (hit_mask * non_zero_mask).sum() / hit_mask.sum()
        ans2 = (hit_mask * non_zero_mask).sum() / non_zero_mask.sum()
        UP =  ((hit_mask * non_zero_mask).sum()/non_zero_mask.sum()) / (non_zero_mask.sum()/len(non_zero_mask))
        print("HR:", ans, "Find Ratio:", ans2)
        # print("UP:", ans/(non_zero_mask.sum()/len(non_zero_mask)))


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
STmodel.load_state_dict(best_model)
torch.save(STmodel,'pth/STZINB_ny_full_5min.pth')
