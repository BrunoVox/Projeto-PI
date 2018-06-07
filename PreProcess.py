import SetParams
from scipy import signal
from skimage.transform import resize
from skimage.color import rgb2ycbcr
import numpy as np

def ImgDownSample(img, blur_kernel, scale):
    if type(img) == int:
        img = float(img)
    if img.shape[2] == 1:
        img_blur = signal.convolve2d(img, blur_kernel, mode="same")
    else:
        img_blur = np.zeros(np.shape(img))
        for i in range(0, img.shape[2]):
            img_blur[:, :, i] = signal.convolve2d(img[:, :, i], blur_kernel, mode="same")
    img_ds = resize(img, (img.shape[0] / scale, img.shape[1] / scale))
    img_ds = int(img_ds)
    return img_ds


def PreProcess(img):
    img_ds = ImgDownSample(img, blur_kernel, sr_scale)
    [number_rows, number_cols, number_channels] = np.shape(img_ds)[0], np.shape(img_ds)[1], np.shape(img_ds)[2]
    if number_channels == 3:
        img_hr = rgb2ycbcr(img)
        img_lr = rgb2ycbcr(img_ds)
        img_hr = float(img_hr[:, :, 0])
        img_lr = float(img_lr[:, :, 0])
    else:
        img_hr = float(img)
        img_lr = float(img)

    nchannels_feat = len(lr_filters)
    img_rs = resize(img_lr, lr_feat_scale)
    [nr, nc] = np.shape(img_rs)[0], np.shape(img_rs)[1]
    feat_lr = np.zeros(nr, nc, nchannels_feat)

    for i in range(0, len(nchannels_feat)):
        feat_lr[:, :, i] = signal.convolve2d(img_rs, lr_filters[i], "same")

    img_data = {"img_ori": img, "img_ds": img_ds, "img_hr": img_hr, "img_lr": img_lr, "feat_lr": feat_lr}
