"""
Main module to train neural networks for channel charting / positioning.

@author: Sueda Taner
"""
#%%
import numpy as np
import torch
import pickle
from matplotlib import pyplot as plt
import copy
import os

from utils.parameter import Parameter
from utils.nn_models import TripletModel, SupervisedModel, SemisupervisedModel, predict
from utils.cc_helpers import plot_chart, evaluate_cc, FeatureExtractor
from utils.loss_helpers import count_wrong_triplets, find_bounding_box

#%% 
# Set the device 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Specify your runID
runID = 'your_runID-triplet'

# Specify the learning type. Three options: 'Supervised', 'Semisupervised', and 'Self- or weakly-supervised'
learning_type = 'Supervised' # uses ground truth positions for training
learning_type = 'Semisupervised' # uses the timestamp-based triplet loss and a small subset of ground truth-labeled samples for training
learning_type = 'Triplet' # uses the timestamp-based triplet loss and/or the received power-based bilateration loss for training

# Weakly-supervised learning parameters: 
# lambda_b : the weight of the bilateration loss against triplet loss
# lambda_box : the weight of the LoS bounding box loss loss against triplet loss
# Set both lambda_b and lambda_box to 0 for only self-supervised learning (using only the triplet loss). The resulting channel chart won't be in real-world coordinates if lambda_.
# Set to a positive number for triplet + bilateration + bounding box loss. The resulting channel chart should be in real-world coordinates.
lambda_b = 0 # set to 10 for self-and weakly supervised method P2, keep it at 0 otherwise
lambda_box = 0 # set to 1 for self-and weakly supervised method P2, keep it at 0 otherwise
bb_src= 'ap' 
box_mode = 1
epoch_Lt_on = 0 # triplet loss is turned ON at this epoch

# Semisupervised learning and affine transform parameter
# The number of ground truth labeled points from the training set
n_ref = 30 # 10000 #int

# Set the number of taps (<=13) of the time domain CSI to use for feature extraction
# This also indirectly sets the network size (see hidden_layer_dims) 
W = 13

# Specify saving, plotting & evaluation preferences
save = 1 # Save training parameters and trained model weights
plot_loss_per_epoch = True
eval_on_training_set = 1 # Evaluate the trained model on the training set
eval_on_test_set = 1 # Evaluate the trained model on the test set
measure_perf = 0 # measure KS and TW-CT for each epoch during training

#%% Parameter setting - you do not need to change these (but you can)

# Specify the ratio of (num_test_samples / num_all_samples)
test_to_all_ratio = 0.2

# Set the feature extraction method
# Options: 'abs', 'reim', 'beamspace abs', 'beamspace reim','cor abs', 'cor reim', 'beamspace + cor abs'
fe_method = 'abs'

# Triplet loss parameters
Tc = 5
M_t = 0.6 # margin in triplet loss
num_triplets_per_anchor = 1

# Bilateration loss parameters
M_p = 3 # dB # margin to pick the "near-far" AP pairs (their powers should differ by at least M_p)
P_thresh = 15 # dB # power threshold - do not consider the APs whose power is less than this (as they are probably nLoS)
M_b = M_t # margin in bilateration loss
# The received power per AP measure to sort the APs
# 0 for computing power, else for genie (as a sanity check)
pow_per_ap_type = 0

# Network size parameters 
num_layers = 5 # number of hidden layers
# Start with the input feature dimension and halven it for num_layers
hidden_layer_dims = lambda input_dim: tuple((input_dim * 1/2**np.arange(num_layers)).astype(np.int32))

# Training parameters
use_scheduler, scheduler_param = 1, 25 # reduce the lr every 25 epochs
num_epochs, batch_size = 100, 100 
lr = 1e-3

# TO MODIFY end=============================================================================
#%% Fix the seed
torch.manual_seed(10)
np.random.seed(10)

#%% Import AP antenna positions, channel matrices, UE positions and timestamps
data_path = 'data/'

ap_pos = np.load(data_path + 'AP_pos.npy')[:,:2].astype(np.float32) 
par0 = Parameter(ap_pos)

H = np.load(data_path + 'H.npy').astype(np.complex64)  # U B W
H = H[:,:,:W]
UE_pos = np.load(data_path + 'UE_pos.npy')[:,:2].astype(np.float32) 
timestamps = np.load(data_path + 'timestamps.npy').astype(np.float32) 

# Box regions
bounding_boxes = np.load(data_path + 'bounding_boxes_APs.npy').astype(np.float32) 

A = ap_pos.shape[0] # the number of APs
Mr = int(H.shape[1] / A) # the number of antennas per AP

# par0 stores the ground truth position of the whole dataset
par0.set_UE_info(UE_pos) # default settings to get the color map for all

#%% Separate the dataset into training and test sets 

# Calculate the number of test samples
n_te = round(par0.U * test_to_all_ratio)
n_tr = par0.U - n_te
# Randomly choose the test sample indices
te_UEs = np.random.choice(par0.U, n_te, False)
                
# Set the training sample indices 
temp = np.arange(par0.U)
tr_UEs = temp[np.isin(temp, te_UEs, invert=True)] # get the UEs that are NOT in test UEs

# Get the training and test samples
H_tr, UE_pos_tr, timestamps_tr = H[tr_UEs], UE_pos[tr_UEs], timestamps[tr_UEs]
H_te, UE_pos_te = H[te_UEs], UE_pos[te_UEs] 

#%% Parameter object for the training set
par_tr = copy.deepcopy(par0)
par_tr.set_UE_info(UE_pos_tr, par0.color_map[tr_UEs], timestamps_tr, tr_UEs)

#%% The power measure for the bilateration loss
if pow_per_ap_type == 0: # Compute received power from the channels
    temp = np.reshape(H_tr, (H_tr.shape[0], A, Mr*H_tr.shape[-1]))
    pow_per_ap = 20*np.log10(np.linalg.norm(temp, ord=2, axis=-1))
    
else: # Use the ground truth positions so that the "power" is inversely proportional to the distance between the UE and an AP
    ap_pos_rpt = par_tr.ap_pos[:,:2].reshape((1, A, 2)).repeat(par_tr.U, 1, 1) # U,A,2
    y_a = (par_tr.UE_pos).reshape((par_tr.U,1,2)) # U,1,2
    dist_per_ap = np.linalg.norm(y_a - np.array(ap_pos_rpt), 2, -1)  # U,A
    pow_per_ap = - 20*np.log10(dist_per_ap) 

err_per_u, num_per_u = count_wrong_triplets(pow_per_ap, par_tr.UE_pos, ap_pos, M_p, P_thresh)
print('For margin', M_p, '\nnum of users with some AP pairs:', np.count_nonzero(num_per_u))
print('num of u-ap-ap triplets:', np.sum(num_per_u))

nz_idcs = np.where(num_per_u != 0)[0]
false_ratio_per_u = err_per_u[nz_idcs] / num_per_u[nz_idcs]
worstidx = np.argmax(false_ratio_per_u)

print('avg of false u-ap-ap triplets:', np.array([np.sum(err_per_u)/np.sum(num_per_u)]))
print('avg num AP pairs per u:', np.array([np.mean(num_per_u)]))

pow_per_ap_n = pow_per_ap - np.max(pow_per_ap, -1, keepdims=True)
pow_per_ap_dB = torch.from_numpy(pow_per_ap)

# Assign bounding box labels based on max powered AP for each UE
box_labels_tr = np.argmax(pow_per_ap, 1)

#%% Torchify variables to prep for training
timestamps_tr = torch.from_numpy(par_tr.UE_timestamps)
UE_pos_tr = torch.from_numpy(par_tr.UE_pos)

#%% Train the model  
fe = FeatureExtractor(fe_method) # Create the FeatureExtractor object
X_tr = fe.feature_extract(torch.from_numpy(H_tr), A) # Extract features from the CSI
hidden_features = hidden_layer_dims(X_tr.shape[1]) # Calculate the hidden layer dimensions according to the input feature dimension

par_tr.fe = fe  # Store which feature extraction method was used
# Store the network parameters in the most extensive way, some of them may go unusued depending on learning_type
par_tr.set_training_params(learning_type=learning_type, W=W, in_features=X_tr.shape[1], hidden_features=hidden_features, out_features=2, 
                   Tc=Tc, Tf=np.inf, M_t=M_t,
                   P_thresh=P_thresh, M_p=M_p, M_b=M_b, lambda_b=lambda_b, lambda_box = lambda_box, bb_src=bb_src, box_mode=box_mode,
                   lr=lr, use_scheduler=use_scheduler, scheduler_param=scheduler_param,
                   num_epochs=num_epochs, batch_size=batch_size, num_triplets_per_anchor=num_triplets_per_anchor,
                   n_ref=n_ref, epoch_L1_on=epoch_Lt_on)
    
if learning_type == 'Triplet':
    train_set = torch.utils.data.TensorDataset(X_tr, timestamps_tr, pow_per_ap_dB, UE_pos_tr)  # can also use time stamps here instead
    model = TripletModel(device, par_tr)
    if lambda_box != 0:
        loss_per_epoch = model.train(train_set, torch.from_numpy(bounding_boxes), box_labels_tr, measure_perf=measure_perf)
    else: loss_per_epoch = model.train(train_set, measure_perf=measure_perf)
        
elif learning_type == 'Supervised':
    train_set = torch.utils.data.TensorDataset(X_tr, UE_pos_tr)  
    model = SupervisedModel(device, par_tr)
    loss_per_epoch = model.train(train_set)

elif learning_type == 'Semisupervised':
    assert n_ref <= par_tr.U
    ref_idcs = np.random.choice(par_tr.U, n_ref)
    train_set = torch.utils.data.TensorDataset(X_tr, timestamps_tr, UE_pos_tr)  # can also use time stamps here instead
    model = SemisupervisedModel(device, par_tr)
    par_tr.ref_pos = par_tr.UE_pos[ref_idcs]
    loss_per_epoch = model.train(train_set, ref_idcs)

print('Training completed!')

if plot_loss_per_epoch:
    plt.figure()
    plt.plot(np.log10(loss_per_epoch))
    plt.xlabel('epoch')
    plt.title('Log10(avg loss of batches in each epoch)')

#%% Find an affine transform (code from \url{https://dichasus.inue.uni-stuttgart.de/tutorials/tutorial/dissimilarity-metric-channelcharting/})
def pad(x: np.array): return np.hstack([x, np.ones((x.shape[0], 1))])
def unpad(x: np.array): return x[:,:-1]
def find_affine_transform(groundtruth_pos, channel_chart_pos):
    A, res, rank, s = np.linalg.lstsq(pad(channel_chart_pos), pad(groundtruth_pos), rcond = None)
    return A
def apply_affine_transform(A: np.array, x: np.array): return unpad(pad(x) @ A)
assert par_tr.n_ref <= par_tr.U
ref_idcs = np.random.choice(par_tr.U, par_tr.n_ref)
affine_transform = find_affine_transform(par_tr.UE_pos[ref_idcs], predict(model, X_tr[ref_idcs]).numpy())

#%% Evaluate and plot
def eval_and_plot(par, model, X, set_str):
    Y = predict(model, X)
    Y = Y.numpy()
    
    ks = evaluate_cc(par.UE_pos, Y, 'KS')
    tw, ct = evaluate_cc(par.UE_pos, Y, 'TW-CT')
    print(f'---\n{set_str} set performance:\nKS:{ks:.3f} TW:{tw:.3f} CT:{ct:.3f}')
    if lambda_b == 0 and lambda_box == 0 and (learning_type == 'Triplet'): 
        plot_chart(Y, par.color_map, False, None, False, title=f'CC-{set_str}. KS={ks:.3f} TW={tw:.3f} CT={ct:.3f}')
        
        Y = apply_affine_transform(affine_transform, Y)
        ks = evaluate_cc(par.UE_pos, Y, 'KS')
        tw, ct = evaluate_cc(par.UE_pos, Y, 'TW-CT')
        
    # Calculate position error
    pos_error = np.linalg.norm(Y-par.UE_pos,2,-1)    
    err_stat = np.around(np.array([np.mean(pos_error), np.median(pos_error), 
                          np.percentile(pos_error, 95), np.amax(pos_error)]), decimals=2)
    print('Mean, median, 95th pctile, max distance error:', err_stat, '\n---')
    
    plot_chart(Y, par.color_map, False, par.ap_pos, 
                     title=f'CC-{set_str}. KS:{ks:.3f} TW:{tw:.3f} MDE={err_stat[0]} 95DE={err_stat[2]}') 
    
if eval_on_training_set:
    eval_and_plot(par_tr, model, X_tr, 'train')
    
#%% Test the trained models
par_te = copy.deepcopy(par_tr)
par_te.set_UE_info(UE_pos_te, par0.color_map[te_UEs])
par_te.dataset_idcs = te_UEs

if eval_on_test_set:
    X = par_te.fe.feature_extract(torch.from_numpy(H_te), A)
    par0.plot_scenario(passive=False, dimensions='2d', title='G.t. pos of test set')
    eval_and_plot(par_te, model, X, 'test')
        
#%% Saving name adjustments
if learning_type == 'Supervised': # triplet parameters or loss weights do not matter
    runID = runID + f'-bs{batch_size}-lr{lr}-{num_epochs}epochs-hidden{hidden_features[0]}'
else:
    if lambda_b == 0 and lambda_box == 0:
        runID = runID + f'-Tc{Tc}-Mt{M_t}-{par_tr.num_triplets_per_anchor}tperanch-bs{batch_size}-lr{lr}-{num_epochs}epochs-hidden{hidden_features[0]}'
    else:
        runID = runID + f'-lambda_b{lambda_b}-lambda_box{lambda_box}-powtype{pow_per_ap_type}-Pthresh{P_thresh}-Mp{M_p}-Tc{Tc}-Mt{M_t}-{par_tr.num_triplets_per_anchor}tperanch-bs{batch_size}-lr{lr}-{num_epochs}epochs-hidden{hidden_features[0]}'
if use_scheduler: runID += f'-schpar{scheduler_param}'

#%% Save the learned network parameters and the training/testing settings
if save:
    output_folder = f'results/network_params'
    os.makedirs(output_folder, exist_ok=True)
    torch.save(model.network.state_dict(), f'{output_folder}/{runID}.pth')
    
    output_folder = f'results/training_testing_params'
    os.makedirs(output_folder, exist_ok=True)
    f = open(f'{output_folder}/{runID}.pckl', 'wb')
    pickle.dump([par_tr, par_te], f)
    f.close()
