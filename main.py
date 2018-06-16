import numpy as np 
import types
from skimage.color import rgb2ycbcr, ycbcr2rgb
from scipy import signal
from math import floor
from random import randint

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def ImgDownSample(img, blur_kernel, scale):
    if type(img) == int:
        img = float(img)
    if img.shape[2] == 1:
        img_blur = signal.convolve2d(img, blur_kernel, mode="same")
    else:
        img_blur = np.zeros((np.shape(img)))
        for i in range(img.shape[2]):
            img_blur[:,:,i] = signal.convolve2d(img[:,:,i], blur_kernel, mode="same")
    img_ds = resize(img, (img.shape[0]/scale, img.shape[1]/scale))
    img_ds = int(img_ds)
    return img_ds

def PreProcess(img, params):
    img_ds = ImgDownSample(img, params.blur_kernel, params.sr_scale)
    # nrow = np.shape(img_ds)[0]
    # ncol = np.shape(img_ds)[1]
    nchl = np.shape(img_ds)[2]
    if nchl == 3:
        img_hr = rgb2ycbcr(img)
        img_lr = rgb2ycbcr(img_ds)
        img_hr = float(img_hr[:,:,0])
        img_lr = float(img_lr[:,:,0])
    else:
        img_hr = float(img)
        img_lr = float(img)

    nchannels_feat = len(params.lr_filters)
    img_rs = resize(img_lr, params.lr_feat_scale)
    nr = np.shape(img_rs)[0]
    nc = np.shape(img_rs)[1]
    feat_lr = np.zeros((nr, nc, nchannels_feat))

    for i in range(len(nchannels_feat)):
        feat_lr[:,:,i] = signal.convolve2d(img_rs, params.lr_filters[i], "same")

    return img, img_ds, img_hr, img_lr, feat_lr

def CollectTrainData(img_data, sample_num, dim_hr, dim_lr, params):
    patch_size_lr = params.patch_data_lr
    nrow = np.shape(img_data.img_lr)[0]
    ncol = np.shape(img_data.img_lr)[1]
    nrlu_lr = np.random.randint(1, high=(nrow - patch_size_lr - 1), size=(1, sample_num))
    nclu_lr = np.random.randint(1, high=(ncol - patch_size_lr - 1), size=(1, sample_num))
    nrlu_hr = nrlu_lr * params.sr_scale
    nclu_hr = nclu_lr * params.sr_scale
    nrlu_lr *= params.lr_feat_scale
    nclu_lr *= params.lr_feat_scale
    patch_size_hr = patch_size_lr * params.sr_scale
    patch_size_lr *= params.lr_feat_scale

    patch_data_hr = PatchExtraction(img_data.img_hr, patch_size_hr, 0, "fixloc", nrlu_hr, nclu_hr)
    #patch_data_hr = bsxfun FAZER MÉTODO
    patch_data_lr = PatchExtraction(img_data.feat_lr, patch_size_lr, 0, "fixloc", nrlu_lr, nclu_lr)
    return np.concatenate(patch_data_hr / dim_hr, patch_data_lr / dim_lr, axis=0)

def Train(params):
    if np.all(params.dict_file != 0):
        open(params.dict_file)
        dict_out = dictionary
    return dict_out

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

def LocalSolve(img_lr, feat_lr, dictionary, params):
    nrow = np.shape(img_lr)[0]
    ncol = np.shape(img_lr)[1]
    nrow_hr = params.sr_scale * nrow
    ncol_hr = params.sr_scale * ncol

    img_out = resize(img_lr, np.shape([nrow_hr, ncol_hr]))
    aggr_times = np.zeros((nrow_hr, ncol_hr))

    return img_out, aggr_times

def GlobalSolve(img_sr, img_lr, params):
    nrow1 = np.shape(img_sr)[0]
    ncol1 = np.shape(img_sr)[1]

    nrow2 = np.shape(img_lr)[0]
    ncol2 = np.shape(img_lr)[1]

    return nrow1, ncol1, nrow2, ncol2, img_sr

def GetSRColorImage(img_sr, img_ds):
    nrow1 = np.shape(img_sr)[0]
    ncol1 = np.shape(img_sr)[1]

    # alguma coisa
    img_us = resize(img_ds, (nrow1,ncol1))
    img_us = rgb2ycbcr(img_us)
    img_sr_min = min(img_sr[:])
    img_sr_max = max(img_sr[:])
    img_sr = (img_sr - img_sr_min) * img_sr_max/(img_sr_max - img_sr_min)
    img_us[:,:,1] = int(img_sr)

    return ycbcr2rgb(img_us)

class params:

    traindata_dir = r"~/Downloads/TrainData"
    test_img = r"~/Downloads/Test/test.jpg"
    sr_scale = 3
    sr_stride = 1
    blur_kernel = matlab_style_gauss2D((3,3),1.2)
    patch_size_lr = 5
    dict_size = 512
    patch_num = 12000
    dict_file = "dict_SR"
    lr_filters = [[1,0,-1], [[1],[0],[-1]], [1,0,-2,0,1], [[1],[0],[-2],[0],[1]]] # ERA TUPLE
    lr_feat_nchl = len(lr_filters)
    lr_feat_scale = 2

    class train_param:

        K = 512
        mode = 3
        lmbd = 10
        iter = 100

    class solve_param:

        mode = 3
        lmbd = 10
        
    glbopt_iter = 10
    glbopt_tau = 0.01
    glbopt_lmbd = 0.1

from skimage.io import imread

img = imread(params.test_img)

nrow = np.shape(img)[0]
ncol = np.shape(img)[1]
nchl = np.shape(img)[2]

img_data = types.SimpleNamespace()
temp = PreProcess(img, params)
img_data.img_ori = temp[0]
img_data.img_ds = temp[1]
img_data.img_hr = temp[2]
img_data.img_lr = temp[3]
img_data.feat_lr = temp[4]

from skimage.transform import resize

img_us = resize(img_data.img_ds, params.sr_scale)

print('Training dictionary...')
dictionary = Train(params.dict_file)

print('Local optimization...')
local_temp = LocalSolve(img_data.img_lr, img_data.feat_lr, dictionary, params)
img_sr = local_temp[0]
aggr_times = local_temp[1]

temp = GlobalSolve(img_sr, img_data.img_lr, params)
img_out = temp[4]
ncol1 = temp[1]
nrow1 = temp[0]
nrow2 = temp[2]
ncol2 = temp[3]
img_sr = img_out # ALGO ESTÁ ESTRANHO

for i in range(1,params.glbopt_iter):
    img_data.img_ds = resize(img_out,[nrow2,ncol2])
    diff1 = img_data.img_ds - img_data.img_lr
    diff2 = img_out - img_sr
    diff1 = resize(diff1,[nrow1,ncol1])
    img_out = img_out - params.glbopt_tau * (diff1 + params.glbopt_lmbd * diff2)

stride = params.sr_stride
patch_size_hr = params.patch_size_lr * params.sr_scale
patch_size_feat = params.patch_size_lr * params.lr_feat_scale
dim_hr = patch_size_hr ** 2
dim_feat = patch_size_feat ** 2 * params.lr_feat_nchl

for nrlu in range(1, (nrow - params.patch_size_lr), stride):

    print('Local Optimizing, row', nrlu)
    nrlu1 = nrlu * params.sr_scale
    nrlu2 = nrlu * params.lr_feat_scale

    for nclu in range(1, (ncol - params.patch_size_lr), stride):
        nclu1 = nclu * params.sr_scale
        nclu2 = nclu * params.lr_feat_scale
        patch_lr = img_data.img_lr[nrlu:nrlu + params.patch_size_lr - 1, nclu:nclu + params.patch_size_lr -1]
        local_mean = np.mean(patch_lr[:])
        patch_hr = img_data.img_out[nrlu1:nrlu1 + patch_size_hr - 1,nclu1:nclu1 + patch_size_hr - 1]
        patch_feat = img_data.feat_lr[nrlu2:nrlu2 + patch_size_feat - 1, nclu2:nclu2 + patch_size_feat - 1]
        aggr_times_local = aggr_times[nrlu1:nrlu1 + patch_size_hr -1,nclu1:nclu1 + patch_size_hr - 1]

        idx_nnz = (patch_hr[:] != 0)
        patch_hr_data = patch_hr[:] - np.mean(patch_hr[idx_nnz])/dim_hr
        patch_feat = patch_feat[:]/dim_feat

        patch_data = [patch_hr_data[idx_nnz],patch_feat]
        dict_temp = np.concatenate((dictionary.dict_hr[idx_nnz,:],dictionary.dict_lr), axis=1)
        
        alpha = spams.OMP(patch_data, dict_temp)# params.solve_param)
        # elif params.train_method == 'lasso':
        #     alpha = spams.Lasso(patch_data, D=dict_temp)
        patch_recov = dictionary.dict_hr * csc_matrix.todense(alpha)
        patch_recov = np.reshape(patch_recov, np.shape([patch_size_hr,patch_size_hr])) * dim_hr + local_mean
        patch_recov = np.divide(np.multiply(patch_recov + patch_hr,aggr_times_local),(1 + aggr_times_local))
        aggr_times_local = aggr_times_local + 1
        aggr_times_local[nrlu1:nrlu1 + patch_size_hr - 1,nclu1:nclu1 + patch_size_hr -1] = aggr_times_local
        img_data.img_out[nrlu1:nrlu1 + patch_size_hr - 1,nclu1:nclu1 + patch_size_hr - 1] = patch_recov

if nchl == 3:
    img_sr = GetSRColorImage(img_sr, img_data.img_lr)

