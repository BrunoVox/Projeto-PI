import numpy as np
from sklearn.transform import resize
#from SetParams import nr

def LocalSolve(img_lr, feat_lr, dictionary):
    [nrow, ncol] = np.shape(img_lr)[0], np.shape(img_lr)[1]
    nrow_hr = sr_scale * nrow
    ncol_hr = sr_scale * ncol
    stride = sr_stride
    patch_size_hr = patch_size_lr * sr_scale
    patch_size_feat = patch_size_lr * lr_feat_scale
    dim_hr = patch_size_hr ** 2
    dim_feat = patch_size_feat ** 2 * lr_feat_nchannel

    img_out = resize(img_lr, [nrow_hr, ncol_hr])