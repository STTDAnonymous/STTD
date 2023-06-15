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
from utils import generate_dataset, get_normalized_adj, get_Laplace, calculate_random_walk_matrix,gauss_loss
from model import *
import random,os,copy
import math
import tqdm
from scipy.stats import truncnorm
import pickle as pk
import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
# Parameters
torch.manual_seed(0)
device = torch.device('cuda') #use_gpu = False
#num_timesteps_input = 24
num_timesteps_output = 12 # num_timesteps_input # 12
num_timesteps_input = num_timesteps_output

space_dim = 100
batch_size = 12
hidden_dim_s = 42
hidden_dim_t = 42
rank_s = 20
rank_t = 4

epochs = 35 #500

# Initial networks
TCN1 = B_TCN(space_dim, hidden_dim_t, kernel_size=6).to(device=device)
TCN2 = B_TCN(hidden_dim_t, rank_t, kernel_size = 6, activation = 'linear').to(device=device)
TCN3 = B_TCN(rank_t, hidden_dim_t, kernel_size= 6).to(device=device)
# TCN4 = B_TCN(hidden_dim_t, space_dim, kernel_size =6, activation = 'linear')
TNB = GaussNorm(hidden_dim_t,space_dim).to(device=device)
SCN1 = D_GCN(num_timesteps_input, hidden_dim_s, 3).to(device=device)
SCN2 = D_GCN(hidden_dim_s, rank_s, 2, activation = 'linear').to(device=device)
SCN3 = D_GCN(rank_s, hidden_dim_s, 2).to(device=device)
# SCN4 = D_GCN(hidden_dim_s, num_timesteps_input, 3, activation = 'linear')
SNB = GaussNorm(hidden_dim_s,num_timesteps_output).to(device=device)
STmodel = ST_Gau(SCN1, SCN2, SCN3, TCN1, TCN2, TCN3, SNB,TNB).to(device=device)

# Load dataset
A = np.load('ny_data_60min/adj_only10_rand0.npy')
X = np.load('ny_data_60min/cta_samp_only10_rand0.npy')
X = X.T
X = X.astype(np.float32)
X = X.reshape((X.shape[0],1,X.shape[1]))
split_line1 = int(X.shape[2] * 0.6)
split_line2 = int(X.shape[2] * 0.7)

print(X.shape,A.shape)
# normalization
max_value = np.max(X.shape[2] * 0.6)
#X = X/max_value
#means = np.mean(X, axis=(0, 2))
#X = X - means.reshape(1, -1, 1)
#stds = np.std(X, axis=(0, 2))
#X = X / stds.reshape(1, -1, 1)

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
A_q = torch.from_numpy((calculate_random_walk_matrix(A_wave).T).astype('float32')).to(device=device)
A_h = torch.from_numpy((calculate_random_walk_matrix(A_wave.T).T).astype('float32')).to(device=device)
A_q = A_q.to(device=device)
A_h = A_h.to(device=device)
# Define the training process
# criterion = nn.MSELoss()
optimizer = optim.Adam(STmodel.parameters(), lr=1e-3)
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
    for i in range(0, training_input.shape[0], batch_size):
        STmodel.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        X_batch = X_batch.to(device=device)
        y_batch = y_batch.to(device=device)

        loc_train,scale_train = STmodel(X_batch,A_q,A_h)
#       print('batch and n',np.mean(X_batch.detach().cpu().numpy()),np.mean(n_train.detach().cpu().numpy()))
#        print(np.mean(n_train.detach().cpu().numpy()))
#        print('ybatchshape',y_batch.shape)
        loss = gauss_loss(y_batch,loc_train,scale_train)
#       print('loss',loss)
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
    training_nll.append(sum(epoch_training_losses)/len(epoch_training_losses))
    ## Step 2, validation
    with torch.no_grad():
        STmodel.eval()
        val_input = val_input.to(device=device)
        val_target = val_target.to(device=device)

        loc_val,scale_val = STmodel(val_input,A_q,A_h)
#        print(n_val)
        val_loss    = gauss_loss(val_target,loc_val,scale_val).to(device="cpu")
        validation_nll.append(np.asscalar(val_loss.detach().numpy()))

        # Calculate the probability mass function for up to 35 vehicles
        #y = range(36)
        #probs = nbinom.pmf(y, n, p)

        # Calculate the expectation value
        a,b = 0,np.inf
        val_pred = truncnorm.mean(a=a,b=b,loc=loc_val.detach().cpu().numpy(),scale=scale_val.detach().cpu().numpy())
        print(val_pred.mean())
        # Calculate the 80% confidence interval
        #lower, upper = nbinom.interval(0.8, n, p)
        
        mae = np.mean(np.abs(val_pred - val_target.detach().cpu().numpy()))
        validation_mae.append(mae)

        n_val,p_val = None,None
        val_input = val_input.to(device="cpu")
        val_target = val_target.to(device="cpu")
    
    print('Epoch %d: trainNLL %.5f; valNLL %.5f; mae %.4f'%(epoch,
    training_nll[-1],validation_nll[-1],validation_mae[-1]))
    if np.asscalar(val_loss.detach().numpy()) == min(validation_nll):
        best_model = copy.deepcopy(STmodel.state_dict())
    checkpoint_path = "checkpoints/"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    with open("checkpoints/losses.pk", "wb") as fd:
        pk.dump((training_nll, validation_nll, validation_mae), fd)

STmodel.load_state_dict(best_model)
torch.save(STmodel,'pth/ST_Truncnorm_ny_60min_samp_only10_in12-out12-h12_nonorm_20210901.pth')
