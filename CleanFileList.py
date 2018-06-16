import numpy as np
from os.path import splitext

def IsReserve(file_name, ext_set):
    flag = 0
    [fpath, fname, fext] = splitext(file_name)[0], 0, splitext(file_name)[1]  # ARRUMAR O NOME DO ARQUIVO OU FNAME = DIR+FILE NAME
    for i in range(1, len(ext_set)):
        if fext == ext_set[i]:
            flag = 1
            break
    return flag, [fpath, fname, fext]

def CleanFileList(file_list, ext_set):
    num_file = len(file_list)
    valid_flag = np.zeros((1, num_file))
    for i in range(1, num_file):
        if (IsReserve(file_list[i].name, ext_set)):
            valid_flag[i] = 1
    return file_list[valid_flag == 1], np.sum(valid_flag)

temp = CleanFileList(file_list, ext_set)
file_list_new = temp [0]
num_file = temp[1]
