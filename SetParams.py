# from skimage.filters import gaussian
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

# class params():
#     class train_param():
#         def K(self):
#             return
#         def mode(self):
#             return
#         def lmbd(self):
#             return
#         def iter(self):
#             return

#     class solve_param():
#         def K(self):
#             return
#         def mode(self):
#             return
#         def lmbd(self):
#             return
#         def iter(self):
#             return

# params = types.SimpleNamespace()

# def SetParams():

class params:

    traindata_dir = r"~/Downloads/TrainData"
    test_img = r"~/Downloads/Test/test.jpg"
    sr_scale = 3
    sr_stride = 1
    blur_kernel = matlab_style_gauss2D((3,3),1.2)
    patch_size_lr = 5
    dict_size = 512
    patch_num = 12000
    dict_file = "dict_SR"
    lr_filters = [[1,0,-1], [[1],[0],[-1]], [1,0,-2,0,1], [[1],[0],[-2],[0],[1]]] # ERA TUPLE
    lr_feat_nchannel = len(lr_filters)
    lr_feat_scale = 2

    class train_param:

        # train_method = "omp"
        # if params.train_method == "omp":
        K = 512
        mode = 3
        lmbd = 10
        iter = 100

    class solve_param:

        mode = 3
        lmbd = 10
        # elif params.train_method == "lasso":
        #     params.train_param.K = params.dict_size
        #     params.train_param.mode = 0
        #     params.train_param._lambda = 0.8
        #     params.train_param.iter = 100
        #     params.solve_param.mode = 0
        #     params.solve_param._lambda = 0.8

    glbopt_iter = 10
    glbopt_tau = 0.01
    glbopt_lmbd = 0.1

    # return params

#train_param = types.SimpleNamespace()

# params_temp = SetParams()
# params = params_temp

