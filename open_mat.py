import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, join as pjoin
# import scipy.io as sio
from scipy.io import loadmat
import pandas as pd

# load the mat files
# data_dir = pjoin(dirname(sio.__file__), 'matlab')
# mat_fname = pjoin(data_dir, 'Squeezenet.mat')
# mat_contents = sio.loadmat(mat_fname)
# mat = sio.matlab.loadmat('Squeezenet.mat')
mat = loadmat('Squeezenet.mat')
# mat = {k:v for k, v in mat.items() if k[0] != '_'}
# data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()})
# data.to_csv('Squeezenet.csv')
print(mat.keys())
x = mat['relu_conv1']
print(x)
print(type(x))
df = pd.DataFrame(x)
df.to_csv('Squeeze.csv')

