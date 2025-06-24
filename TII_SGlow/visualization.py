import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import math


# args
map_root = ''
Preds = np.load('./output_f4_16/SGlow/Preds.npy')
Reals = np.load('./output_f4_16/SGlow/Reals.npy')

K = 20
observed = 16

# show
mean_true = Reals.mean(axis=(0, 1), keepdims = True)[:, :, :2]
std_true = Reals.std(axis=(0, 1), keepdims = True)[:, :, :2]
if Preds.ndim ==4:
    mean_true = mean_true[np.newaxis,:, :, :2]
    std_true = std_true[np.newaxis, :, :, :2]
    Preds = Preds[:,:,:,:2] * std_true + mean_true
else:
    Preds = Preds[:, :, :2] * std_true + mean_true

Reals = Reals[:,:,:2]

keep_ship = [7,9,11,14,16, 17,25,21,28,29,41,43,44,46,50,56,57] #+ [k for k in range(40,50)]

plt.scatter(Reals[:, keep_ship, 0], Reals[:, keep_ship, 1],c='b',s=3)
if Preds.ndim ==4:
    plt.scatter(Preds[:, keep_ship, :,0], Preds[:, keep_ship, :,1],c='r',alpha=0.02, s=20)
else:
    plt.scatter(Preds[:, keep_ship,  0], Preds[:, keep_ship,  1], c='r', alpha=0.8, s=3)



imp = plt.imread(map_root)
plt.imshow(imp, extent=[114.099003, 114.187537, 22.265695, 22.322062])
plt.show()
print(';')

