import numpy as np
from SetParams import dict_file, traindata_dir, patch_num, sr_scale, patch_size_lr, lr_feat_scale, lr_feat_nchannel
from math import floor
from skimage.io import imread_collection
from CleanFileList import CleanFileList
from PreProcess import PreProcess
from PatchExtraction import PatchExtraction
from SetParams import train_param

def CollectTrainData(img_data, sample_num, dim_hr, dim_lr):
    [nrow, ncol] = np.shape(img_data[img_lr])
    nrlu_lr = np.random.randint(1, high=(nrow - patch_size_lr - 1), size=(1, sample_num))
    nclu_lr = np.random.randint(1, high=(ncol - patch_size_lr - 1), size=(1, sample_num))
    nrlu_hr = nrlu_lr * sr_scale
    nclu_hr = nclu_lr * sr_scale
    nrlu_lr *= lr_feat_scale
    nclu_lr *= lr_feat_scale
    patch_size_hr *= sr_scale
    patch_size_lr *= sr_scale

    patch_data_hr = PatchExtraction(img_data{key=img_hr}, patch_size_hr, 0, "fixloc", nrlu_hr, nclu_hr)
    patch_data_lr = PatchExtraction(img_data{key=feat_lr}, patch_size_lr, 0, "fixloc", nrlu_lr, nclu_lr)
    collect_data = np.concatenate(patch_data_hr / dim_hr, patch_data_lr / dim_lr, axis=0)


def Train():
    if np.all(dict_file != 0):
        open(dict_file)

    [traindata_list, num_train_img] = CleanFileList((traindata_dir), {".png", ".bmp", ".jpg", ".jpeg"})
    sample_num = floor(patch_num / num_train_img)
    patch_num = sample_num * num_train_img

    patch_size_hr = patch_size_lr * sr_scale
    patch_size_lr = patch_size_lr * lr_feat_scale
    dim_hr = patch_size_hr ** 2
    dim_lr = patch_size_lr ** 2 * lr_feat_nchannel
    train_data = np.zeros((dim_lr + dim_hr, patch_num))

    for i in range(1, len(num_train_img)):
        img = imread_collection([train_data, traindata_list[i].name])
        img_data = PreProcess(img)
        train_data_sub = CollectTrainData(img_data, sample_num, dim_hr, dim_lr)
        train_data[:, (i - 1) * sample_num + 1 : i * sample_num] = train_data_sub

    dictonary = mexTrainDL(train_data, train_param)
    dict_out.dict_hr = dictionary(1 : dim_hr, :)
    dict_out.dict_lr = dictionary(dim_hr + 1 : end, :)
    

    