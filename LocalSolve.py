import numpy as np
from skimage.transform import resize
from PreProcess import PreProcess, params
#from SetParams import nr
from scipy.sparse import csc_matrix

def LocalSolve(img_lr, feat_lr, dictionary, params):
    nrow = np.shape(img_lr)[0]
    ncol = np.shape(img_lr)[1]
    nrow_hr = params.sr_scale * nrow
    ncol_hr = params.sr_scale * ncol
    stride = params.sr_stride
    patch_size_hr = patch_size_lr * params.sr_scale
    patch_size_feat = patch_size_lr * params.lr_feat_scale
    dim_hr = patch_size_hr ** 2
    dim_feat = patch_size_feat ** 2 * params.lr_feat_nchl

    img_out = resize(img_lr, [nrow_hr, ncol_hr])
    aggr_times = np.zeros((nrow_hr, ncol_hr))

    return img_out, aggr_times

for nrlu in range(1, (nrow - patch_size_lr), stride):
    print('Local Optimizing, row', nrlu)
    nrlu1 = nrlu * params.sr_scale
    nrlu2 = nrlu * params.lr_feat_scale

    for nclu in range(1, (ncol - patch_size_lr), stride):
        nclu1 = nclu * params.sr_scale
        nclu2 = nclu * params.lr_feat_scale
        patch_lr = img_lr[nrlu:nrlu + patch_size_lr - 1, nclu:nclu + patch_size_lr -1]
        local_mean = np.mean(patch_lr[:])
        patch_hr = img_out[nrlu1:nrlu1 + patch_size_hr - 1,nclu1:nclu1 + patch_size_hr - 1]
        patch_feat = feat_lr[nrlu2:nrlu2 + patch_size_feat - 1, nclu2:nclu2 + patch_size_feat - 1]
        aggr_times_local = aggr_time[nrlu1:nrlu1 + patch_size_hr -1,nclu1:nclu1 + patch_size_hr - 1]

        idx_nnz = (patch_hr[:] != 0)
        patch_hr_data = patch_hr[:] - np.mean(patch_hr[idx_nnz])/dim_hr
        patch_feat = patch_feat[:]/dim_feat

        patch_data = [patch_hr_data[idx_nnz],patch_feat]
        dict_temp = np.concatenate((dictionary.dict_hr[idx_nnz,:],dictionary.dict_lr), axis=1)
        
        if params.train_method == 'omp':
            alpha = spams.OMP(patch_data, dict_temp)# params.solve_param)
        elif params.train_method == 'lasso':
            alpha = spams.Lasso(patch_data, D=dict_temp)
        patch_recov = dictionary.dict_hr * csc_matrix.todense(aplha)
        patch_recov = np.reshape(patch_recov, np.shape([patch_size_hr,patch_size_hr])) * dim_hr + local_mean
        patch_recov = np.divide(np.multiply((patch_recov + patch_hr,aggr_times_local),(1 + aggr_times_local)))
        aggr_times_local = aggr_times_local + 1
        aggr_times_local[nrlu1:nrlu1 + patch_size_hr - 1,nclu1:nclu1 + patch_size_hr -1] = aggr_times_local
        img_out[nrlu1:nrlu1 + patch_size_hr - 1,nclu1:nclu1 + patch_size_hr - 1] = patch_recov
