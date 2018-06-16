from skimage.filters import gaussian
import numpy as np
import types

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

class params():
    class train_param():
        def K(self):
            return
        def mode(self):
            return
        def lmbd(self):
            return
        def iter(self):
            return

    class solve_param():
        def K(self):
            return
        def mode(self):
            return
        def lmbd(self):
            return
        def iter(self):
            return

def SetParams():
    params.traindata_dir = r"C:\Users\bruno\OneDrive\Documents\ProjetosCC\Projeto PI\TrainData"
    params.test_img = r"C:\Users\bruno\OneDrive\Documents\ProjetosCC\Projeto PI\TestData\tree.jpg"
    params.sr_scale = 3
    params.sr_stride = 1
    params.blur_kernel = matlab_style_gauss2D((3,3),1.2)
    params.patch_size_lr = 5
    params.dict_size = 512
    params.patch_num = 12000
    params.dict_file = "dict_SR"
    params.lr_filters = [[1,0,-1], [[1],[0],[-1]], [1,0,-2,0,1], [[1],[0],[-2],[0],[1]]] # ERA TUPLE
    params.lr_feat_nchannel = len(params.lr_filters)
    params.lr_feat_scale = 2

    params.train_method = "omp"
    if params.train_method == "omp":
        params.train_param.K = params.dict_size
        params.train_param.mode = 3
        params.train_param.lmbd = 10
        params.train_param.iter = 100
        params.solve_param_mode = 3
        params.solve_param_lambda = 10
    elif params.train_method == "lasso":
        params.train_param.K = params.dict_size
        params.train_param.mode = 0
        params.train_param._lambda = 0.8
        params.train_param.iter = 100
        params.solve_param.mode = 0
        params.solve_param._lambda = 0.8

    params.glbopt_iter = 10
    params.glbopt_tau = 0.01
    params.glbopt_lambda = 0.1

    return params

#train_param = types.SimpleNamespace()
params_temp = types.SimpleNamespace()
params_temp = SetParams()
params = params_temp