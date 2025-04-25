"""
Define a custom object whose attributes are channel scenario and training parameters. 

@author: Sueda Taner
"""
#%%
import numpy as np
import torch
from matplotlib import pyplot as plt
import os
from pathlib import Path

#%% Create Parameter object

class Parameter:
    def __init__(self, ap_pos=None, desired_max_SNR=25):
        # Initialize the Parameter object with the AP positions
        self.ap_pos = ap_pos
        # The UEs' information will be set by method set_UE_info
        
        self.max_SNRdB = desired_max_SNR  # a number or np.inf
        
    # Training-related settings
    def set_training_params(self, learning_type=None, W=16, in_features=256, hidden_features=(256,128,64), out_features=2, 
                   Tc=10, Tf=np.inf, segment_start_idcs=[0], M_t=1,
                   P_thresh=np.inf, M_p=0, M_b=1, lambda_b=0, lambda_box=0,
                   box_mode=None, bb_src=None,
                   lr=1e-3, use_scheduler=0, scheduler_param=None,
                   num_epochs=50, batch_size=100,
                   num_triplets_per_anchor=1, epoch_L1_on=0, n_ref=0):
        
        self.learning_type = learning_type
        self.W = W # Number of time domain channel taps used for feature extraction   
        
        # Network dimension parameters
        self.in_features = in_features  # must match the number of features in dataset
        # For each entry, create a hidden layer with the corresponding number of units
        self.hidden_features = hidden_features
        self.out_features = out_features
            
        # Triplet loss parameters
        self.Tc = Tc # samples that are at most this far in time are positive samples
        self.Tf = Tf  # samples that are at least Tc and at most this far in time are negative samples       
        self.segment_start_idcs = segment_start_idcs # segment start indices of the dataset 
        self.M_t = M_t # margin in triplet loss
        
        # Bilateration loss parameters
        self.P_thresh = P_thresh # APs whose power is less than ((the best AP of the user) - P_thresh) are assumed nLoS and ignored
        self.M_p = M_p # (dB) APs whose powers differ by at least this muchh can be assumed nearer/further
        self.M_b = M_b # margin in bilateration loss
        self.lambda_b = lambda_b # the weight of the bilateration loss against the triplet loss
        
        # Training parameters
        self.learning_rate = lr # training learning rate
        self.use_scheduler = use_scheduler # use learning rate scheduler
        self.scheduler_param = scheduler_param # lr schedueler parameter: reduce the lr at every this many epochs
        self.num_epochs = num_epochs # number of training epochs 
        self.batch_size = batch_size # training batch size. take this many anchors (but the number of triplets can be larger)
        # Rarely changed parameters:
        self.num_triplets_per_anchor = num_triplets_per_anchor # for a batch, randomly pick this many triplets for each anchor
        self.epoch_L1_on = epoch_L1_on # turn ON triplet loss from this epoch on   
        
        self.lambda_box = lambda_box
        if lambda_box:
            assert box_mode is not None and bb_src is not None
        self.box_mode = box_mode
        self.bb_src = bb_src
        
        self.n_ref = n_ref
        
    # Simulation scenario-related settings    
    def set_UE_info(self, UE_pos, color_map=None, UE_timestamps=None, dataset_idcs=None):
        self.UE_pos = UE_pos
        self.U = UE_pos.shape[0]
        if UE_timestamps is None:
            self.UE_timestamps = torch.arange(self.U)
        else:
            self.UE_timestamps = UE_timestamps
        if color_map is None:
            self.set_default_color_map()  # set the color map
        else:
            self.color_map = color_map
        self.dataset_idcs = dataset_idcs

    def set_default_color_map(self):
        # Coloring of the users
        color1 = (self.UE_pos[:, 0] - np.min(self.UE_pos[:, 0])).reshape((self.U, 1)) \
                 / (np.max(self.UE_pos[:, 0]) - np.min(self.UE_pos[:, 0]))
        color2 = (self.UE_pos[:, 1] - np.min(self.UE_pos[:, 1])).reshape((self.U, 1)) \
                 / (np.max(self.UE_pos[:, 1]) - np.min(self.UE_pos[:, 1]))
        color3 = np.zeros((self.U, 1))

        self.color_map = np.concatenate((color1, color2, color3), axis=-1)

    def plot_scenario(self, passive=False, dimensions='2d', color_map=None, title=None):
        if color_map is None:
            color_map = self.color_map
        fig = plt.figure()
        if dimensions == '2d':
            ax = fig.add_subplot()
            # ax.scatter(self.UE_pos[:, 0], self.UE_pos[:, 1], marker='o', c=self.color_map)
            ax.scatter(self.UE_pos[:, 0], self.UE_pos[:, 1], s=5, c=color_map)
            
            ax.scatter(self.ap_pos[:, 0], self.ap_pos[:, 1], marker='^')
            for a in range(self.ap_pos.shape[0]):
                ax.annotate(a, (self.ap_pos[a, 0], self.ap_pos[a, 1]))
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.axis('equal')
            # ax.set_aspect('equal', 'box')
            ax.grid()
        else:
            raise Exception('Undefined dimensions for plotting the scenario')
        if title is not None:
            plt.title(title)
    
    def add_noise_np(self, H: np.array) -> np.array:
        """
        Add noise so that the max SNR per UE-AP pair is fixed to self.max_SNRdB.
        Let H channel matrix of an AP where the rows correspond to antennas and columns to delay taps/subcarriers
        SNR_dB = 10 log10 (Frobenius_norm(H)**2 / (N0 * size(H)))
        
        Do not add noise to the exact zeros in the channel (as we would not have the CSI for those in the first place)

        Parameters
        ----------
        H : np.array
            Channel matrix of size num_users U, num_total_antennas B, num_subcarriers/delay taps W.
        
        Returns
        -------
        Hn : np.array
            Noisy channel matrix.

        """
        A = self.ap_pos.shape[0]
        assert H.shape[1] % A == 0
        Mr = int(H.shape[1] / A )
        
        SNR = 10**(self.max_SNRdB / 10)
        
        T = np.reshape(H, (H.shape[0], A, Mr*H.shape[-1]))
        pow_per_ap = np.linalg.norm(T, ord=2, axis=-1)
        N0 = np.max(pow_per_ap)**2 / T.shape[-1] / SNR 
        N = np.sqrt(N0 / 2) * (np.random.randn(*H.shape)
                               + 1j * np.random.randn(*H.shape))
        
        s = pow_per_ap
        n = np.linalg.norm(np.reshape(N, (H.shape[0], A, Mr*H.shape[-1])), 2, -1)
        arr = 20*np.log10(s[s != 0]/n[s != 0]) # per AP SNRs
        arr2 = 10*np.log10(s[s != 0]**2/(N0*(T.shape[-1])))
        print('Actual SNR per AP', np.min(arr[arr != - np.inf]), np.max(arr), np.mean(arr[arr != - np.inf]))
        print('Expected SNR per AP', np.min(arr2[arr2 != - np.inf]), np.max(arr2), np.mean(arr2[arr2 != - np.inf]))
        
        fig = plt.figure()
        data_sorted = np.sort(arr[arr != - np.inf])
        cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
        
        plt.plot(data_sorted, cdf)
        plt.xlabel("SNR")
        plt.ylabel("cumulative probability")
        plt.grid()
        
        # Keep the zeros as zeros
        zero_idcs = np.where(H == 0)
        Hn = H + N.astype(np.complex64)
        Hn[zero_idcs] = 0
        
        return Hn
