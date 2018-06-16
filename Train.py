import numpy as np
from PreProcess import img_data
from SetParams import params#.dict_file, params.traindata_dir, params.patch_num, params.sr_scale, params.patch_size_lr, params.lr_feat_scale, params.lr_feat_nchannel
from math import floor
from skimage.io import imread#_collection
from CleanFileList import CleanFileList
from PreProcess import PreProcess
from PatchExtraction import PatchExtraction
# from SetParams import train_param
import spams
import types

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
    
traindata_dir = params.traindata_dir
patch_num = params.patch_num
sr_scale = params.sr_scale

[traindata_list, num_train_img] = CleanFileList((traindata_dir), {".png", ".bmp", ".jpg", ".jpeg"})
sample_num = floor(patch_num/num_train_img)
patch_num = sample_num * num_train_img

patch_size_hr = params.patch_size_lr * sr_scale
patch_size_lr = params.patch_size_lr * params.lr_feat_scale
dim_hr = patch_size_hr ** 2
dim_lr = patch_size_lr ** 2 * params.lr_feat_nchl
train_data = np.zeros((dim_lr + dim_hr, patch_num))

for idx_img in range(1, len(num_train_img)):
    img = imread([params.train_data, traindata_list[idx_img].name])
    img_data = PreProcess(img, params)
    train_data_sub = CollectTrainData(img_data, sample_num, dim_hr, dim_lr, params)
    train_data[:,(idx_img - 1) * sample_num + 1 : idx_img * sample_num] = train_data_sub

dict_out = types.SimpleNamespace()
dictionary = spams.TrainDL(train_data, params.train_param)
dict_out.dict_hr = dictionary[1:dim_hr,:]
dict_out.dict_lr = dictionary[dim_hr + 1:len(dictionary),:]
