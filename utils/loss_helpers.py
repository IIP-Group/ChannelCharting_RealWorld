"""
Helper functions for the triplet loss and AP receive power-related losses.

@author: Sueda Taner
"""
#%%
import numpy as np
import torch

#%% Bounding box loss-related

def find_bounding_box(pos, boxes):
    """
    pos : N, 2
    boxes: N_box, 2, 2. Nonoverlapping rectangular boxes defined by two corners: xmin, ymin; xmax, ymax

    """
    labels = - np.ones(len(pos))
    for u, pos_u in enumerate(pos):
        in_box_x = np.logical_and(pos_u[0] >= boxes[:,0,0], pos_u[0] <= boxes[:,1,0])
        in_box_y = np.logical_and(pos_u[1] >= boxes[:,0,1], pos_u[1] <= boxes[:,1,1])
        in_box = np.logical_and(in_box_x, in_box_y)
        if in_box.any():
            labels[u] = np.where(in_box)[0][0]
    
    return labels

def bb_loss(pos, boxes):
    """
    Bounding box loss.
    If the RX power at a certain AP is high, the UE must be in the LoS area of this AP.
    Suppose I have a (2,2) np.array for the LoS area of each AP. # [x1 y1; xend yend]
    First - use genie boxes. One box per UE
    
    pos N, 2
    boxes N, 2, 2
    """
    
    out_box = torch.logical_or(pos[:,0] < boxes[:,0,0], pos[:,0] > boxes[:,1,0])
    x_penalty = torch.sum(torch.minimum(torch.abs(pos[out_box,0]-boxes[out_box,0,0]), 
                                        torch.abs(pos[out_box,0]-boxes[out_box,1,0]))**2, -1)
    
    out_box = torch.logical_or(pos[:,1] < boxes[:,0,1], pos[:,1] > boxes[:,1,1])
    y_penalty = torch.sum(torch.minimum(torch.abs(pos[out_box,1]-boxes[out_box,0,1]), 
                                        torch.abs(pos[out_box,1]-boxes[out_box,1,1]))**2, -1)
    
    return (x_penalty + y_penalty) / pos.shape[0]
    
#%% Bilateration loss-related
def get_los_aps_per_u(pow_per_ap, thresh):
    valid_APs_per_u = [[]]*pow_per_ap.shape[0]
    num_valid_APs_per_u = np.zeros(pow_per_ap.shape[0], dtype=int)
    for u in range(pow_per_ap.shape[0]):
        valid_APs = torch.arange(len(pow_per_ap[u]),dtype=int)[pow_per_ap[u] > thresh]
        valid_APs_per_u[u] = valid_APs
        num_valid_APs_per_u[u] = len(valid_APs)
    return valid_APs, num_valid_APs_per_u

def count_wrong_triplets(pow_per_ap, UE_pos, ap_locs, M_p=0, P_thresh=-np.inf, false_threshold=0):
    # !pow_per_ap in dB
    A = ap_locs.shape[0]
    U = UE_pos.shape[0]
    
    num_wrong_per_user = np.zeros((U), dtype=int)
    num_ap_pairs_per_user = np.zeros((U), dtype=int)
    ap_pairs = [[]]*U
    valid_APs_per_u = [[]]*U
    num_valid_APs_per_u = np.zeros(U, dtype=int)
    # We won't include APs whose received power is < (the best AP's power - P_thresh)
    # The AP power values are normalized so that the max is 0 dB for each UE
    for u in range(U):
        valid_APs = np.arange(A, dtype=int)[pow_per_ap[u] > P_thresh]
        valid_APs_per_u[u] = valid_APs
        num_valid_APs_per_u[u] = len(valid_APs)
        if num_valid_APs_per_u[u] > 0:
            x = pow_per_ap[u][:,None]
            v = x[valid_APs] - x[valid_APs].T
            # The AP powers should differ by > M_p for a valid AP pair
            w = np.where(v > M_p)
            if len(w[0]) != 0:
                ap_pairs[u] = np.vstack((valid_APs[w[0]], valid_APs[w[1]])) 
                num_ap_pairs_per_user[u] = ap_pairs[u].shape[1]
                dist_near = np.linalg.norm(UE_pos[u] - ap_locs[valid_APs[w[0]]], 2, -1)
                dist_far  = np.linalg.norm(UE_pos[u] - ap_locs[valid_APs[w[1]]], 2, -1)
                num_wrong_per_user[u] = np.sum(dist_near - dist_far > false_threshold) 
    return num_wrong_per_user, num_ap_pairs_per_user
    
#%% Triplet loss-related
def get_pos_neg_edges(all_timestamps, Tc, Tf, segment_start_idcs=[0]):
    """
    Given time stamps, Tc, Tf, and segments, create the pos - neg sample
    interval edges for each anchor.

    Parameters
    ----------
    all_timestamps : arraylike
        Timestamps of all datapoints. MUST be in ascending order within each segment.
    Tc : float
        The samples that are at most Tc apart in time are positive (near).
    Tf : float
        The samples that are at least Tc and at most Tf apart in time are negative (far).
    segment_start_idcs : arraylike
        Segment start indices of the dataset so that the triplets can be 
        formed only from the same segment. The default is [0].

    Returns
    -------
    anchors : np.array of size (num_anchors,)
        Indices of anchor samples.
    pos_edges : np.array of size (num_anchors, 2, 2)
        Start and end indices of positive samples to the left and right of an anchor.
    neg_edges : np.array of size (num_anchors, 2, 2)
        Start and end indices of negative samples to the left and right of an anchor.

    """
    eps = 1e-14 # small number to avoid precision issues
    n = len(all_timestamps)
    assert n > segment_start_idcs[-1]

    all_timestamps = np.array(all_timestamps) # Make sure it's np
    
    num_sections = len(segment_start_idcs)
    idcs = list(range(n))
    anchors = []
    pos_edges_for_each_anchor = []
    neg_edges_for_each_anchor = []
    for section_idx, segment_start_idx in enumerate(segment_start_idcs):
        # Work with the time stamps for each segment
        if section_idx != num_sections - 1:  # if not the last segment
            timestamps = all_timestamps[segment_start_idx:segment_start_idcs[section_idx + 1]]
        else:  # if the last segment
            timestamps = all_timestamps[segment_start_idx:]
        n_section = len(timestamps)
        idcs = list(np.arange(n_section) + segment_start_idx)
        for idx_in_section, t0 in enumerate(
                timestamps):
            i = idx_in_section + segment_start_idx
            
            close_idcs = np.nonzero(np.abs(timestamps - t0) <= Tc + eps)[0] + segment_start_idx
            
            far_idcs = np.setdiff1d((np.nonzero(np.abs(timestamps - t0) <= Tf + eps)[0] + segment_start_idx),
                                    close_idcs, assume_unique=True)
            
            close_idcs = np.setdiff1d(close_idcs, np.array([i]), assume_unique=True)

            if close_idcs.size != 0 and far_idcs.size != 0:
                anchors.append(i)
                pos_area_edges = []
                neg_area_edges = []

                if close_idcs[0] > i or close_idcs[-1] < i:
                    num_pos_area = 1 # either left or right
                    pos_area_edges = np.tile(np.array([[close_idcs[0], close_idcs[-1] + 1]]), (2,1))
                else:
                    # num_pos_area = 2 # both left and right
                    pos_area_edges = np.array([[close_idcs[0], i],
                                               [i + 1, close_idcs[-1] + 1]])
                if far_idcs[0] > i or far_idcs[-1] < i:
                    num_neg_area = 1 # either left or right
                    neg_area_edges = np.tile(np.array([[far_idcs[0], far_idcs[-1] + 1]]), (2,1))
                else:
                    # num_neg_area = 2 # both left and right
                    neg_area_edges = np.array([[far_idcs[0], min(close_idcs[0], i)],
                                               [max(i + 1, close_idcs[-1]), far_idcs[-1] + 1]])
                pos_edges_for_each_anchor.append(pos_area_edges)
                neg_edges_for_each_anchor.append(neg_area_edges)
                
    anchors = np.array(anchors)
    A = len(anchors)
    pos_edges, neg_edges = np.zeros((A, 2, 2), dtype=np.int32), np.zeros((A, 2, 2), dtype=np.int32)
    for a in range(A): 
        pos_edges[a] = pos_edges_for_each_anchor[a]
        neg_edges[a] = neg_edges_for_each_anchor[a]
    return anchors, pos_edges, neg_edges

def get_triplet_batch(anchors, pos_edges, neg_edges, num_triplets_per_anchor=1):
    """
    Given pos and neg sample intervals (edges) for each anchor, randomly choose the pos
    and neg samples to generate a batch of triplets.

    Parameters
    ----------
    anchors : np.array
        Indices of anchor points.
    pos_edges : np.array of size (num_anchors,2,2)
        The interval of pos samples to the left and right of an anchor point.
        If all pos samples are on one side of anchor i, pos_edges[i,0] = pos_edges[i,1].
    neg_edges : np.array of size (num_anchors,2,2)
        The interval of neg samples to the left and right of an anchor point.
        If all neg samples are on one side of anchor i, neg_edges[i,0] = neg_edges[i,1].
    num_triplets_per_anchor : int
        Number of random triplets to generate per anchor. The default is 1.

    Returns
    -------
    triplet_batch : np.array of size (num_anchors, num_triplets_per_anchor, num_triplets_per_anchor)
        Batch of triplets.

    """
    num_anchors = len(anchors)
    # We want the same interval to apply to all pos-neg samples of the same anchor, and
    # np.random.randint(low, high, (m, n)) requires len(low) = len(high) = n

    # Randomly choose if the positive samples are from the left (1) or right (0)
    is_pos_left = np.random.randint(0,2, (num_triplets_per_anchor, num_anchors))
    # Generate random numbers from the interval of pos samples
    pos = (is_pos_left * np.random.randint(pos_edges[:,0,0], pos_edges[:,0,1], (num_triplets_per_anchor, num_anchors))
    + (1-is_pos_left) * np.random.randint(pos_edges[:,1,0], pos_edges[:,1,1], (num_triplets_per_anchor, num_anchors)))
    pos = pos.T.flatten()
    
    # Randomly choose if the negative samples are from the left (1) or right (0)
    is_neg_left = np.random.randint(0,2, (num_triplets_per_anchor, num_anchors))
    # Generate random numbers from the interval of neg samples
    neg = (is_neg_left * np.random.randint(neg_edges[:,0,0], neg_edges[:,0,1], (num_triplets_per_anchor, num_anchors))
    + (1-is_neg_left) * np.random.randint(neg_edges[:,1,0], neg_edges[:,1,1], (num_triplets_per_anchor, num_anchors)))
    neg = neg.T.flatten()
    
    triplet_batch = np.zeros((num_anchors * num_triplets_per_anchor, 3), dtype=np.int32)
                    
    triplet_batch[:, 0] = np.repeat(anchors, num_triplets_per_anchor)
    triplet_batch[:, 1] = pos
    triplet_batch[:, 2] = neg

    return triplet_batch
