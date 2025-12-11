#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:16:57 2024
Affine transform helpers.
"""
#%%
import numpy as np

#%%
def pad(x: np.array): return np.hstack([x, np.ones((x.shape[0], 1))])
def unpad(x: np.array): return x[:,:-1]
def find_affine_transform(groundtruth_pos, channel_chart_pos):
    A, res, rank, s = np.linalg.lstsq(pad(channel_chart_pos), pad(groundtruth_pos), rcond = None)
    return A
def apply_affine_transform(A: np.array, x: np.array): return unpad(pad(x) @ A)
