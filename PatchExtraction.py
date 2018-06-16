from math import floor
import numpy as np
from random import randint

def PatchExtraction(img, patch_size, pad_flag, crop_type, *args):
    img_pad = img
    patch_size_half = floor(patch_size/2)
    if pad_flag == 1:
        img_pad = np.pad(img, ([patch_size_half,patch_size_half]), 'reflect')
    nrow = np.shape(img_pad)[0]
    ncol = np.shape(img_pad)[1]
    nchl = np.shape(img_pad)[2]

    nrow_lu_max = nrow - patch_size + 1
    ncol_lu_max = ncol - patch_size + 1

    if crop_type == 'rand':
        patch_num = args[1]
        patch_data = np.zeros((patch_size ** 2 * nchl,patch_num))
        for idx_patch in range(1, patch_num):
            nrow_lu = randint(1, nrow_lu_max)
            ncol_lu = randint(1, ncol_lu_max)
            cur_patch = img_pad[nrow:nrow_lu + patch_size - 1,ncol_lu:ncol_lu + patch_size - 1,:]
            patch_data[:,idx_patch] = cur_patch[:]
        
    elif crop_type == 'regular':
        stride = args[1]
        nrow_lu = (nrow_lu_max - 1)/stride[1] # VER SE ESTÁ CERTO
        ncol_lu = (ncol_lu_max - 1)/stride[2] # VER SE ESTÁ CERTO
        patch_num = len(nrow_lu) * len(ncol_lu)
        patch_data = np.zeros((patch_size * patch_size * nchl, patch_num))
        for i in range(1, len(nrow_lu)):
            for j in range(1, len(ncol_lu)):
                idx_patch = j + (i - 1) * len(ncol_lu)
                cur_patch = img_pad[nrow_lu[i]:nrow_lu[i] + patch_size - 1,ncol_lu[j]:ncol_lu[j] + patch_size - 1,:]
                patch_data[:,idx_patch] = cur_patch[:]

    elif crop_type == 'fixloc':
        nrow_lu = args[1]
        ncol_lu = args[2]
        if len(nrow_lu) != len(ncol_lu):
            print('nrow_lu and ncol_lu must have the same length')
        patch_num = len(nrow_lu)
        patch_data = np.zeros((patch_size ** 2 * nchl, patch_num))
        for i in range(1, len(patch_num)):
            cur_patch = img_pad[nrow_lu[i]:nrow_lu[i] + patch_size - 1, ncol_lu[i]:ncol_lu[i] + patch_size - 1,:]
            patch_data[:,i] = cur_patch[:]

    else:
        print('Unkown Crop Type')

    return patch_data

#patch_data = PatchExtraction(img, patch_size, pad_flag, crop_type)