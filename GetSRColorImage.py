import numpy as np
from PreProcess import img_ds
from skimage.transform import resize
from skimage.color import rgb2ycbcr, ycbcr2rgb

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

img_out = GetSRColorImage(img_sr, img_ds)
