#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 17:54:13 2024

@author: taners
"""
#%%
import torch
import numpy as np
from matplotlib import pyplot as plt

#%%
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

def reorder_out2_in_segments(h0_w0_list, n_seg):
    reordering = []
    segment_separation_idcs = np.concatenate((np.array([0]), np.cumsum(n_seg)))        
    # for seg_idx, (idx_st, idx_end) in enumerate(zip(segment_start_idcs[:-1], segment_start_idcs[1:])): # buggy version that somehow gives better results...
    for seg_idx, (idx_st, idx_end) in enumerate(zip(segment_separation_idcs[:-1], segment_separation_idcs[1:])):
        reordering.append(reorder_out2in(*h0_w0_list[seg_idx]) + idx_st)
        # UE_pos[idx_st:idx_end] = UE_pos[reordering]
        # H[idx_st:idx_end] = H[reordering]
    return np.concatenate(reordering) 
   
def reorder_out2in(h,w):
    n = w * h  
    X1 = np.arange(n).astype(np.int32)
    X2 = np.reshape(X1,(h, w))
    
    idx = 0
    count = 0
    cur_h, cur_w = h, w
    arr = np.array([]).astype(np.int32)
    while cur_h > 0 and  cur_w > 0:
        # corner0 = idx
        # print(idx)
        corner1 = idx + cur_w - 1
        corner2 = corner1 + (cur_h-1)*w 
        corner3 = corner2 - (cur_w-1)
        temp1 = np.arange(idx, corner1, 1).astype(np.int32)
        arr = np.concatenate((arr, temp1))
        if len(arr) == n: break
        temp2 = np.arange(corner1, corner2, w).astype(np.int32)
        arr = np.concatenate((arr, temp2))
        if len(arr) == n: break
        if cur_h > 1:
            temp3 = np.arange(corner2, corner3, -1).astype(np.int32)
            arr = np.concatenate((arr, temp3)) 
        else:
            arr = np.concatenate((arr, np.array([corner2]).astype(np.int32)))
            break
        if len(arr) == n: break
        if cur_w > 1:
            temp4 = np.arange(corner3,idx,-w).astype(np.int32)
            arr = np.concatenate((arr, temp4))
        else:
            arr = np.concatenate((arr, np.array([corner3]).astype(np.int32)))
            break
        
        cur_h, cur_w  = cur_h - 2, cur_w - 2
        count += 1
        idx += w + 1
    return arr

#%%
if __name__ == '__main__':
    h,w = 153,104
    h,w = 5,8
    n = h*w
    x = reorder_out2in(h,w)
    print(x)
        
#%%    

def to_flip_odd_rows(h, w):
    n = w * h  
    X1 = np.arange(n).astype(np.int32)
    X2 = np.reshape(X1,(h, w))
    # print(X2)
    X3 = X2
    X3[1::2,:] = np.flip(X3[1::2,:], -1)
    # print(X3)
    X4 = X3.flatten()
    return X4

def to_flip_odd_columns(h,w, reverse=True):
    n = w * h  
    X1 = np.arange(n).astype(np.int32)
    X2 = np.reshape(X1,(h, w))
    X3 = X2
    X3[:,1::2] = np.flip(X3[:,1::2], 0)
    # print('before reverse:\n', X3)
    if reverse: 
        X3 = X3[:,-1::-1]
        # X3 = X3[-1::-1]
        # print('after reverse:\n', X3)    
    X4 = X3.T
    X5 = X4.flatten()
    return X5
#%%
# h,w = 3,5
# a = (to_flip_odd_rows(h,w))
# b = (to_flip_odd_columns(h,w))
# c = np.concatenate((a, -b-1))
# print(c)

# h,w = 2,5
# a = (to_flip_odd_rows(h,w))
# b = (to_flip_odd_columns(h,w))
# c = np.concatenate((a, b[::-1]))
# print(c)

# h,w = 3,2
# a = (to_flip_odd_rows(h,w))
# b = (to_flip_odd_columns(h,w))
# # b = (to_flip_odd_columns(h,w, False))
# c = np.concatenate((a, b))
# print(c)
