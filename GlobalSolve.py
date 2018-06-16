from skimage.transform import resize

def GlobalSolve(img_sr, img_lr, params):
    nrow1 = np.shape(img_sr)[0]
    ncol1 = np.shape(img_sr)[1]

    nrow2 = np.shape(img_lr)[0]
    ncol2 = np.shape(img_lr)[1]

    return nrow1, ncol1, nrow2, ncol2, img_sr

temp = GlobalSolve(img_sr, img_lr, params)
img_out = temp[4]
ncol1 = temp[1]
nrow1 = temp[0]
nrow2 = temp[2]
ncol2 = temp[3]

for i in range(1,params.glbopt_iter):
    img_ds = resize(img_out,[nrow2,ncol2])
    diff1 = img_ds - img_lr
    diff2 = img_out - img_sr
    diff1 = resize(diff1,[nrow1,ncol1])
    img_out = img_out - params.glbopt_tau * (diff1 + params.glbopt_lambda * diff2)
    