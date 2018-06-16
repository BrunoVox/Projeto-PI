import numpy as np
from SetParams import params
from Train import Train
from LocalSolve import LocalSolve
from GlobalSolve import GlobalSolve
from GetSRColorImage import GetSRColorImage
from skimage.io import imread
from skimage.transform import resize

#params = SetParams()
img = imread(params.test_img)
nrow = np.shape(img)[0]
ncol = np.shape(img)[1]
nchl = np.shape(img)[2]
# img_data = PreProcess(img, params)
from PreProcess import img_data
img_us = resize(img_data.img_ds, params.sr_scale)



print('Training dictionary...')
dictionary = Train(params)

print('Local optimization...')
img_sr = LocalSolve(img_data.img_lr, img_data.feat_lr, dictionary, params)
img_sr = GlobalSolve(img_sr, img_data.img_lr, params)

if nchl == 3:
    img_sr = GetSRColorImage(img_sr, img_data.img_lr)
