#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for channel charting in real-world coordinates.

This script implements channel charting in real-world coordinates [1] using
channel-state information (CSI) from the 5G physical uplink shared channel
(PUSCH). The method combines triplet loss with bilateration loss to learn a
channel chart that is grounded in real-world coordinates.

Workflow:
1. Load preprocessed CSI dataset (from gen_dataset.py)
2. Extract features and compute receive power estimates for bilateration loss
3. Prepare training/test splits
4. Train neural network with triplet + bilateration loss
5. Evaluate channel chart quality and save results

This code is a branch of the repository
https://github.com/IIP-Group/ChannelCharting_RealWorld used in [2],
and adapted for the CAEZ-5G-OUTDOOR dataset in [3].

Created on Fri Feb 23 13:21:40 2024
@author: Sueda Taner, edited by Reinhard Wiesmayr

References:
[1] S. Taner, V. Palhares, and C. Studer, "Channel charting in real-world 
    coordinates with distributed MIMO," IEEE Trans. Wireless Commun., 
    vol. 24, no. 9, pp. 7286–7300, 2025.

[2] R. Wiesmayr, F. Zumegen, S. Taner, C. Dick, and C. Studer, "CSI-based user 
    positioning, channel charting, and device classification with an NVIDIA 5G 
    testbed," in Asilomar Conf. Signals, Syst., Comput., Oct. 2025.
"""
#%%
import numpy as np
import torch
import pickle
from matplotlib import pyplot as plt
import copy
import os
import argparse

import scipy as sp
import pandas as pd

from utils.parameter import Parameter
from utils.nn_models import TripletModel, SupervisedModel, SemisupervisedModel, predict
from utils.cc_helpers import plot_chart, evaluate_cc, FeatureExtractor
from utils.loss_helpers import count_wrong_triplets
from utils.segment_processing import find_bounding_box

###############################################################################
# Configuration
###############################################################################
# Specifying the data and results paths
parser = argparse.ArgumentParser(description='Channel charting in real-world coordinates using CAEZ-5G datasets')
parser.add_argument('--data_path', type=str, default='~/csi_data/',
                    help='Path to the directory containing CSI data files (default: ~/csi_data/)')
parser.add_argument('--results_path', type=str, default='~/results/cc/',
                    help='Path to the directory containing results files (default: ~/results/cc/)')
args = parser.parse_args()
data_path = args.data_path
results_path = args.results_path

# Specify path to dataset for random partitioning and (optional) additional test data (e.g., last 500 samples)
data_path_rand_partitioning = data_path + "robot_measurement_outdoor_2025_10_11_auto_correlation_no_norm_truncated_25_sorted_4_rx_ant_wo_last_500.npz"
data_path_additional_test = data_path + "robot_measurement_outdoor_2025_10_11_auto_correlation_no_norm_truncated_25_sorted_4_rx_ant_only_last_500.npz"

# Specify your runID (used for saving results)
runID = '5g_cc_tiplet_box_billat_autocor_robot_measurement_outdoor_2025_10_11_eval_last_500_1'

# Modify the parameters below to configure training

# Set the device (GPU/CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Specify the learning type
# Options: 'Supervised', 'Semisupervised', 'Triplet'
# - 'Supervised': uses ground truth positions for training
# - 'Semisupervised': uses triplet loss + small subset of ground truth-labeled samples
# - 'Triplet': uses timestamp-based triplet loss + bilateration loss
learning_type = 'Triplet'

# Loss function parameters
# lambda_b: weight of bilateration loss against triplet loss
#   Set to 0: use only triplet loss (channel chart not in real-world coordinates)
#   Set to positive: use triplet + bilateration loss (channel chart in real-world coordinates)
lambda_b = 10  # Non-zero value enables real-world coordinate grounding
lambda_box = 1  # Weight for bounding box loss
bb_src = 'ap'  # Bounding box source: 'ap' (AP-based) or 'genie' (ground-truth)
box_mode = 1
epoch_Lt_on = 0  # Epoch at which triplet loss is turned on (0 = from start)

# Semisupervised learning parameter
n_ref = 50  # Number of ground-truth labeled points for affine transform

# Feature extraction parameter
W = 50  # Number of delay-domain CSI taps to use (matches feature dimension from gen_dataset.py)

# Output and evaluation preferences
save = 1  # Save training parameters and trained model weights
plot_loss_per_epoch = True  # Plot training loss curve
eval_on_training_set = 0  # Evaluate on training set
eval_on_test_set = 1  # Evaluate on test set
measure_perf = 0  # Measure KS and TW-CT metrics during training (slower)

###############################################################################
# Advanced Parameters
###############################################################################
# These parameters typically don't need to be changed

# Specify the ratio of (num_test_samples / num_all_samples)
test_to_all_ratio = 0.2

# Set the feature extraction method
# Other options: 'abs', 'reim', 'beamspace abs', 'beamspace reim','cor abs', 'cor reim', 'beamspace + cor abs'
fe_method = 'none' #  'none' because we take in directly features

# Triplet loss parameters
Tc = 1.5 # seconds for how long the samples are considered close
Tf = 20.0 # seconds for how long the samples are considered far
M_t = 0.3 # margin in triplet loss       # look at equation, @TODO play with this, **1** might be good or even too much, bc my area is small
num_triplets_per_anchor = 2   #  @TODO adjusts the "true" batch size

# Bilateration loss parameters
M_p = 13 # dB # margin to pick the "near-far" AP pairs (their powers should differ by at least M_p)
P_thresh = -np.inf # dB # power threshold - do not consider the APs whose power is less than this (as they are probably nLoS)
M_b = 1.0 # margin in bilateration loss
# The received power per AP measure to sort the APs
# 0 for computing power, else for genie
pow_per_ap_type = 0 # 1

# Network size parameters 
num_layers = 5 # number of hidden layers
# Start with the input feature dimension and halven it for num_layers
hidden_layer_dims = lambda input_dim: tuple((input_dim * 1/2**np.arange(num_layers)).astype(np.int32))

# Training parameters
use_scheduler, scheduler_param = 1, 50
num_epochs, batch_size = 200, 256
lr = 1e-3
# TO MODIFY end=============================================================================
###############################################################################
# Random Seed
###############################################################################
# Set random seeds for reproducibility
torch.manual_seed(10)
np.random.seed(10)

###############################################################################
# Data Loading
###############################################################################
# Load preprocessed CSI dataset (generated by gen_dataset.py)
# The dataset should contain autocorrelation features in delay domain
data = np.load(data_path_rand_partitioning, allow_pickle=True) # Dataset for random partitioning to training / test data
data_ = np.load(data_path_additional_test, allow_pickle=True) # Additional test dataset (e.g., last 500 samples)

# ORU to cell-id map
oru_pos_dict = data["oru_pos_dict"].item()
oru_pos_list = [oru_pos_dict["position_data_oru7_cell41"], oru_pos_dict["position_data_oru8_cell42"], oru_pos_dict["position_data_oru5_cell43"], oru_pos_dict["position_data_oru6_cell51"]] # outdoor caez (fixed/correct)

# fix oru-pos for CAEZ-5G-BASEMENT measurement
# oru_pos_list = [oru_pos_dict["position_data_oru6_cell41"]+[0.0,0.0,0.7], # todo measure height
#                 oru_pos_dict["position_data_oru7_cell42"]+[0.,0.0,0.74], 
#                 oru_pos_dict["position_data_oru8_cell43"]+[0.81,0.0,0.], 
#                 oru_pos_dict["position_data_oru5_cell51"]+[-0.4,0.0,0.]] # basement caez (fixed/correct)

ap_pos = np.array(oru_pos_list).astype(np.float32)
ap_pos = ap_pos[:,[0,2]]
par0 = Parameter(ap_pos)

noise_var_db = data["noise_var"].astype(np.float32)
H_max_sc = data["H_max_sc"].astype(np.float32)

H = data["H"].astype(np.float32)   # U B W
# stack AP and antenna dimension
H_shape = np.shape(H)
H = np.reshape(H, newshape=(H_shape[0], H_shape[1]*H_shape[2], H_shape[3]))

H_ = data_["H"].astype(np.float32)   # U B W
# stack AP and antenna dimension
H_shape_ = np.shape(H_)
H_ = np.reshape(H_, newshape=(H_shape_[0], H_shape_[1]*H_shape_[2], H_shape_[3]))

UE_pos = data["pos"].astype(np.float32) 
UE_pos_ = data_["pos"].astype(np.float32) 
timestamps = (data["timestamps"]-data["timestamps"][0]).astype(np.float32) 

# Box regions
# bounding_boxes_genie = np.load(data_path + 'bounding_boxes_genie.npy').astype(np.float32) 
bounding_boxes_ap = data["bounding_box"]

A = ap_pos.shape[0] # the number of APs
Mr = int(H.shape[1] / A) # the number of antennas per AP

bounding_boxes_genie = bounding_boxes_ap
#######

#%% Assign the correct boxes
box_labels_genie = find_bounding_box(UE_pos, bounding_boxes_genie)

bb_src = 'ap'
if bb_src == 'genie':
    bounding_boxes = bounding_boxes_genie 
elif bb_src == 'ap':
    bounding_boxes = bounding_boxes_ap 

# par0 stores the ground truth position of the whole dataset
par0.set_UE_info(UE_pos) # default settings to get the color map for all
# par0.plot_scenario(passive=False, dimensions='2d', title='Ground truth positions of the whole set')

###############################################################################
# Dataset Splitting
###############################################################################
# Randomly partition dataset into training (80%) and test (20%) sets

# Calculate the number of test samples
n_te = round(par0.U * test_to_all_ratio)
n_tr = par0.U - n_te

# Randomly choose the test sample indices
te_UEs = np.random.choice(par0.U, n_te, False)

# Set the training sample indices (all samples not in test set)
temp = np.arange(par0.U)
tr_UEs = temp[np.isin(temp, te_UEs, invert=True)]

# Get the training and test samples
H_tr, UE_pos_tr, timestamps_tr = H[tr_UEs], UE_pos[tr_UEs], timestamps[tr_UEs]
H_te, UE_pos_te = H[te_UEs], UE_pos[te_UEs] 

noise_var_db_tr = noise_var_db[tr_UEs]
noise_var_db_te = noise_var_db[te_UEs]
H_max_sc_tr = H_max_sc[tr_UEs]
H_max_sc_te = H_max_sc[te_UEs]
###############################################################################
# Training Set Parameter Object
###############################################################################
# Create parameter object for training set with timestamps and color mapping
par_tr = copy.deepcopy(par0)
par_tr.set_UE_info(UE_pos_tr, par0.color_map[tr_UEs], timestamps_tr, tr_UEs)

###############################################################################
# Receive Power Computation (for Bilateration Loss)
###############################################################################
# Compute receive power per AP for bilateration loss
# This is required for real-world coordinate grounding (lambda_b > 0)
if pow_per_ap_type == 0:  # Compute received power from CSI
    # temp = np.reshape(H_tr, (H_tr.shape[0], A, Mr*H_tr.shape[-1]))

    # for autocorrelation features:
    temp = np.reshape(H_tr, (H_tr.shape[0], A, Mr, -1)).take(indices=[0], axis=-1).squeeze(axis=-1) # assuming delay-domain auto-correlation features, index [0] of IFFT is sum over all subcarriers
    pow_per_ap = 10*np.log10(np.linalg.norm(temp, ord=2, axis=-1)) # 10* because autocorrelation features are already squared in frequency domain
    
    # # make power zero mean (compensate variable Rx gain of O-RUs)
    pow_per_ap = pow_per_ap - np.mean(pow_per_ap, axis=0)
    
else: # Use the ground truth positions so that the "power" is inversely proportional to the distance between the UE and an AP
    ap_pos_rpt = np.repeat(par_tr.ap_pos[:,:2].reshape((1, A, 2)), par_tr.U, axis=0) # U,A,2
    y_a = (par_tr.UE_pos).reshape((par_tr.U,1,2)) # U,1,2
    dist_per_ap = np.linalg.norm(y_a - np.array(ap_pos_rpt), 2, -1)  # U,A
    pow_per_ap = - 20*np.log10(dist_per_ap) 

err_per_u, num_per_u = count_wrong_triplets(pow_per_ap, par_tr.UE_pos, ap_pos, M_p, P_thresh)
print('For margin', M_p, 'num of users with some AP pairs:', np.count_nonzero(num_per_u))
print('num of u-ap-ap:', np.sum(num_per_u))

nz_idcs = np.where(num_per_u != 0)[0]
false_ratio_per_u = err_per_u[nz_idcs] / num_per_u[nz_idcs]
worstidx = np.argmax(false_ratio_per_u)
# print(M_p, err_per_u, num_per_u)
print('avg of false u-ap-ap triplets:', np.array([np.sum(err_per_u)/np.sum(num_per_u)]), 
        'worst user:', false_ratio_per_u[worstidx], 'out of ', num_per_u[nz_idcs][worstidx], 'AP pairs.',
        'avg num AP pairs per u:', np.array([np.mean(num_per_u)]))

pow_per_ap_n = pow_per_ap - np.max(pow_per_ap, -1, keepdims=True)
pow_per_ap_dB = torch.from_numpy(pow_per_ap)

if bb_src == 'genie':
    box_labels_tr = box_labels_genie[tr_UEs]
elif bb_src == 'ap':
    box_labels_tr = np.argmax(pow_per_ap, 1)

#%% Torchify variables to prep for training
timestamps_tr = torch.from_numpy(par_tr.UE_timestamps)
UE_pos_tr = torch.from_numpy(par_tr.UE_pos)

###############################################################################
# Feature Extraction and Model Training
###############################################################################
# Extract features from CSI and train the channel charting network

fe = FeatureExtractor(fe_method)  # Create the FeatureExtractor object
X_tr = fe.feature_extract(torch.from_numpy(H_tr), A)  # Extract features from CSI
# manually write dimensions that Sueda is also using
hidden_features = hidden_layer_dims(192) # Calculate the hidden layer dimensions according to the input feature dimension
# hidden_features = hidden_layer_dims(X_tr.shape[1])

segment_start_idcs=[100,X_tr.shape[0]-200]

par_tr.fe = fe  # Store which feature extraction method was used
# Store the network parameters in the most extensive way, some of them may go unusued depending on learning_type
par_tr.set_training_params(learning_type=learning_type, W=W, in_features=X_tr.shape[1], hidden_features=hidden_features, out_features=2, 
                   Tc=Tc, Tf=Tf, M_t=M_t, segment_start_idcs=segment_start_idcs,
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

###############################################################################
# Evaluation Functions
###############################################################################
# Helper functions for evaluating channel chart quality
# Affine transform is used to align channel chart with ground-truth coordinates
# (only needed for triplet-only training when lambda_b=0)

def pad(x: np.array): return np.hstack([x, np.ones((x.shape[0], 1))])
def unpad(x: np.array): return x[:,:-1]

def find_affine_transform(groundtruth_pos, channel_chart_pos):
    """Find affine transformation to align channel chart with ground-truth positions."""
    A, res, rank, s = np.linalg.lstsq(pad(channel_chart_pos), pad(groundtruth_pos), rcond=None)
    return A

def apply_affine_transform(A: np.array, x: np.array): 
    """Apply affine transformation to channel chart positions."""
    return unpad(pad(x) @ A)

# Compute affine transform using reference points (for triplet-only training)
assert par_tr.n_ref <= par_tr.U
ref_idcs = np.random.choice(par_tr.U, par_tr.n_ref)
affine_transform = find_affine_transform(par_tr.UE_pos[ref_idcs], predict(model, X_tr[ref_idcs]).numpy())

def eval_and_plot(par, model, X, set_str, X_=None):
    Y = predict(model, X)
    Y = Y.numpy()
    if X_ is not None:
        Y_ = predict(model, X_)
        Y_ = Y_.numpy()
    else:
        Y_ = None
    
    ks = evaluate_cc(par.UE_pos, Y, 'KS')
    tw, ct = evaluate_cc(par.UE_pos, Y, 'TW-CT')
    print(f'---\n{set_str} set performance:\nKS:{ks:.3f} TW:{tw:.3f} CT:{ct:.3f}')
    if lambda_b == 0 and lambda_box == 0 and (learning_type == 'Triplet'): 
        plot_chart(Y, par.color_map, False, None, False, title=f'CC-{set_str}. KS={ks:.3f} TW={tw:.3f} CT={ct:.3f}', X_=Y_)
        Y = apply_affine_transform(affine_transform, Y)
        ks = evaluate_cc(par.UE_pos, Y, 'KS')
        tw, ct = evaluate_cc(par.UE_pos, Y, 'TW-CT')
        
    # else: 
    # Calculate position error
    pos_error = np.linalg.norm(Y-par.UE_pos,2,-1)    
    err_stat = np.around(np.array([np.mean(pos_error), np.median(pos_error), 
                          np.percentile(pos_error, 95), np.amax(pos_error)]), decimals=2)
    print('Mean, median, 95th pctile, max distance error:', err_stat, '\n---')
    
    box_labels_est = find_bounding_box(Y, bounding_boxes_genie)
    true_segs = np.count_nonzero(box_labels_est == box_labels_genie[par.dataset_idcs])
    
    acc = np.round(true_segs/par.U, decimals=2)
    print('Correctly estimated segments for te:', acc)
    plot_chart(Y, par.color_map, False, par.ap_pos, 
                     title=f'CC-{set_str}. KS:{ks:.3f} TW:{tw:.3f} Acc={acc:.2f} MDE={err_stat[0]} 95DE={err_stat[2]}', X_=Y_) 
    
if eval_on_training_set:
    eval_and_plot(par_tr, model, X_tr, 'train')
    
###############################################################################
# Evaluation
###############################################################################
# Evaluate the trained model on test set

par_te = copy.deepcopy(par_tr)
par_te.set_UE_info(UE_pos_te, par0.color_map[te_UEs])
par_te.dataset_idcs = te_UEs

if eval_on_test_set:
    X = par_te.fe.feature_extract(torch.from_numpy(H_te), A)
    ax = par0.plot_scenario(passive=False, dimensions='2d', title='G.t. pos of test set')
    if H_ is not None:
        X_ = par_te.fe.feature_extract(torch.from_numpy(H_), A)
        ax.scatter(UE_pos_[:, 0], UE_pos_[:, 1], s=5, c='b')
    else:
        X_ = None
    eval_and_plot(par_te, model, X, 'test', X_=X_)
        
###############################################################################
# Save Results
###############################################################################
# Save trained model weights and training parameters
if learning_type == 'Supervised': # triplet parameters or loss weights do not matter
    runID = runID + f'-bs{batch_size}-lr{lr}-{num_epochs}epochs-hidden{hidden_features[0]}'
else:
    if lambda_b == 0 and lambda_box == 0:
        runID = runID + f'-Tc{Tc}-Tf{Tf}-Mt{M_t}-{par_tr.num_triplets_per_anchor}tperanch-bs{batch_size}-lr{lr}-{num_epochs}epochs-hidden{hidden_features[0]}'
    else:
        runID = runID + f'-lambda_b{lambda_b}-lambda_box{lambda_box}-powtype{pow_per_ap_type}-Pthresh{P_thresh}-Mp{M_p}-Mb{M_b}-Tc{Tc}-Tf{Tf}-Mt{M_t}-{par_tr.num_triplets_per_anchor}tperanch-bs{batch_size}-lr{lr}-{num_epochs}epochs-hidden{hidden_features[0]}_epoch_Lt_on{epoch_Lt_on}'
if use_scheduler: runID += f'-schpar{scheduler_param}'

print(runID)

if save:
    # set_model_device(model, "cpu")
    torch.save(model.network.state_dict(), results_path + f'network_params/{runID}.pth')
    
    f = open(results_path + f'training_testing_params/{runID}.pckl', 'wb')
    pickle.dump([par_tr, par_te], f)
    f.close()
#%%    
def save_all_plots(filename, figs=None):
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for idx,fig in enumerate(figs):
        fig.savefig(f'{filename}/plot{idx}.png')  
if save: 
    output_folder = results_path + f'figures/{runID}'
    os.makedirs(output_folder, exist_ok=True)
    save_all_plots(output_folder)

