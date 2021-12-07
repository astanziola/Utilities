import h5py
import numpy as np

def read_matlab_v7(filepath):
    arrays = {}
    f = h5py.File(filepath)
    for k, v in f.items():
        arrays[k] = np.array(v)
    return arrays