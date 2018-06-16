from SetParams import params
from scipy import signal
from skimage.transform import resize
from skimage.color import rgb2ycbcr
import numpy as np
import types

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

img_data = types.SimpleNamespace()
from SR import img
# img_ds = ImgDownSample(img, blur_kernel, scale)
temp = PreProcess(img, params)
img_data.img_ori = temp[0]
img_data.img_ds = temp[1]
img_data.img_hr = temp[2]
img_data.img_lr = temp[3]
img_data.feat_lr = temp[4]