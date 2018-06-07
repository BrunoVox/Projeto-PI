from math import ceil
import numpy as np

def ImgPad(img_in, pad_size, op_type, *args):
    [nr, nc] = np.shape(img_in)[0], np.shape(img_in)[1]
    pad_size_half = ceil(pad_size / 2)
    if op_type == 0:
        if args[0] == 0:
            pad_type = "symmetric"
    
        elif args[0] == 1:
            pad_type = "replicate"
    
        elif args[0] == 2:
            pad_type = "replicate"

        img_pad = np.pad(img_in, (pad_size_half, pad_size_half), "reflect")
    
    elif op_type == 1:
        img_pad = img_in#(pad_size_half + 1 : nr - pad_size_half, pad_size_half + 1 : nc - pad_size_half) CORRIGIR
    else:
        print("Unknown operation type")
    return img_pad