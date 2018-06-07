from skimage.filters import gaussian

class train_param():
    def K(self):
        return
    def mode(self):
        return
    def lmbd(self):
        return
    def iter(self):
        return

traindata_dir = r"C:\Users\bruno\OneDrive\Documents\ProjetosCC\Projeto PI\TrainData"
test_img = r"C:\Users\bruno\OneDrive\Documents\ProjetosCC\Projeto PI\TestData\tree.jpg"
sr_scale = 3
sr_stride = 1
blur_kernel = gaussian(img, sigma=1)
patch_size_lr = 5
dict_size = 512
patch_num = 12000
dict_file = "dict_SR"
lr_filters = {[1, 0, -1], [[1], [0], [-1]], [1, 0, -2, 0, 1], [[1], [0], [-2], [0], [1]]}
lr_feat_nchannel = len(lr_filters)
lr_feat_scale = 2

train_method = "omp"
if train_method == "omp":
    train_param.K = dict_size
    train_param.mode = 3
    train_param.lmbd = 10
    train_param.iter = 100
    solve_param_mode = 3
    solve_param_lambda = 10
if train_method == "lasso":
    train_param.K = dict_size
    train_param.mode = 0
    train_param._lambda = 0.8
    train_param.iter = 100
    solve_param.mode = 0
    solve_param._lambda = 0.8

glbopt_iter = 10
glbopt_tau = 0.01
glbopt_lambda = 0.1