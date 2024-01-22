################################################################################
# Official code implementation of the paper
# "Error estimates for POD-DL-ROMs: a deep learning framework for reduced order
#  modeling of nonlinear parametrized PDEs enhanced by proper orthogonal 
#  decomposition" 
# https://arxiv.org/abs/2305.04680
#
# -> Utilities <-
# 
# Authors:     S.Brivio, S.Fresca, N.R.Franco, A.Manzoni 
# Affiliation: MOX Laboratory (Politecnico di Milano, Mathematics Department)
################################################################################

import numpy as np
np.random.seed(1)

import os 

import scipy.io as sio



def set_checkpoint_folder(self):
    checkpoint_folder = os.path.join(
        self.save_folder, self.name + "_checkpoints"
    )
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    self.checkpoint_filepath = os.path.join(checkpoint_folder, self.name \
        + "_weights.h5")


def loadmat(filename, id : str):
    return sio.loadmat(filename)[id].astype('float32')


def loadnpy(filename):
    return np.load(filename, allow_pickle = True).astype('float32')


def loadfile(filename, id : str):
    if filename.endswith('.npy'):
        filecontent = loadnpy(filename)
    elif filename.endswith('.mat'):
        filecontent = loadmat(filename, id)
    else:
        raise ValueError('Unrecognised file extension') 
    return filecontent
    

class Normalizer:

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.x_min = np.min(self.x_train, axis = 0)
        self.x_max = np.max(self.x_train, axis = 0)
        self.y_min = np.min(self.y_train)
        self.y_max = np.max(self.y_train)

    def forward_x(self, x):
        return (x - self.x_min) / (self.x_max - self.x_min)
    
    def forward_y(self, y):
        return (y - self.y_min) / (self.y_max - self.y_min)
    
    def backward(self, y):
        return self.y_min + y * (self.y_max - self.y_min)
