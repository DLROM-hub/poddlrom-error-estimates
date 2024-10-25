################################################################################
# Official code implementation of the paper
# "Error estimates for POD-DL-ROMs: a deep learning framework for reduced order
#  modeling of nonlinear parametrized PDEs enhanced by proper orthogonal 
#  decomposition" 
# https://doi.org/10.1007/s10444-024-10110-1.
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
    """ Sets the save folder and paths for model checkpoints.
    """
    checkpoint_folder = os.path.join(
        self.save_folder, self.name + "_checkpoints"
    )
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    self.checkpoint_filepath = os.path.join(checkpoint_folder, self.name \
        + "_weights.h5")



def loadmat(filename, id : str):
    """ Loads and converts to float32 a MAT file.

    Args:
        filename: the full filepath.
        id (str): the matrix identifier.
    
    Returns:
        The loaded matrix.
    """
    return sio.loadmat(filename)[id].astype('float32')



def loadnpy(filename):
    """ Loads and converts to float32 a NPY file.

    Args:
        filename: the full filepath.
    
    Returns:
        The loaded array.
    """
    return np.load(filename, allow_pickle = True).astype('float32')



def loadfile(filename, id : str = None):
    """ Frontend to load files with.

    Args:
        filename: the full filepath.
        id (str): the matrix identifier.
    
    Returns:
        The loaded matrix.

    Raises: 
        ValueError: if the file extension is not NPY nor MAT.
    """
    if filename.endswith('.npy'):
        filecontent = loadnpy(filename)
    elif filename.endswith('.mat'):
        filecontent = loadmat(filename, id)
    else:
        raise ValueError('Unrecognised file extension') 
    return filecontent
    




class Normalizer:
    """ It is used to normalize input and output data.
    """

    def __init__(self, x_train, y_train):
        """

        Args:
            x_train: the training inputs.
            y_train: the training outputs.

        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_min = np.min(self.x_train, axis = 0)
        self.x_max = np.max(self.x_train, axis = 0)
        self.y_min = np.min(self.y_train)
        self.y_max = np.max(self.y_train)



    def forward_x(self, x):
        """ Forward pass for inputs.

        Args:
            x: input data
        
        Returns:
            The normalized input data.
        """
        return (x - self.x_min) / (self.x_max - self.x_min)
    


    def forward_y(self, y):
        """ Forward pass for targets.

        Args:
            y: target data
        
        Returns:
            The normalized target data.
        """
        return (y - self.y_min) / (self.y_max - self.y_min)
    


    def backward(self, y):
        """ Backward pass ("de-normalization") for outputs.

        Args:
            y: output data

        Returns:
            The normalized output data.
        """
        return self.y_min + y * (self.y_max - self.y_min)
