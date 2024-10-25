################################################################################
# Official code implementation of the paper
# "Error estimates for POD-DL-ROMs: a deep learning framework for reduced order
#  modeling of nonlinear parametrized PDEs enhanced by proper orthogonal 
#  decomposition" 
# https://doi.org/10.1007/s10444-024-10110-1.
#
# -> Data generator <-
# 
# Authors:     S.Brivio, S.Fresca, N.R.Franco, A.Manzoni 
# Affiliation: MOX Laboratory (Politecnico di Milano, Mathematics Department)
################################################################################

import numpy as np
np.random.seed(1)
from scipy.linalg import svd
from sklearn.utils.extmath import randomized_svd

import utils





class DataGen:
    """ Data generator class: retains train, validation and test set
    """

    def __init__(
        self, 
        pod_dim : int = None, 
        alpha_train :float = 0.8, 
        rpod : bool = True
    ):
        """ 

        Args:
            pod_dim (int): the POD reduced dimension (defaults to None). If None
                           the POD matrix and relative quantities are not 
                           computed.
            alpha_train (float): fraction of the data to allocate for training.
                                 The rest is splitted among validation and test.
                                 (defaults to 0.8).
            rpod (bool): if True, we use randomized POD to compute the POD     
                         matrix
                
        """
        self.pod_dim = pod_dim
        if self.pod_dim is not None:
            self.rpod = rpod
        assert ((alpha_train > 0.0) and (alpha_train < 1.0))
        self.alpha_train = alpha_train
        self.data = dict()
        self.data['train'] = dict()
        self.data['val'] = dict()
        self.data['test'] = dict()



    def read(
        self, 
        filenames : dict, 
        filenames_test : dict = None, 
        transpose : bool = False, 
        N_t : int = 1
    ):
        """ Reads data from given files

        Args:
            filenames (dict): the filenames to read data from. 
            filenames_test (dict): the (optional) test filenames (if test data 
                                   are provided in a different file). (defaults
                                   to None).
            transpose (bool): if True, we transpose the read array (defaults to
                              False).
            N_t (int): The number of timesteps (must be equal for all samples, 
                       defaults to 1).
        """
        self.N_t = N_t
        data = dict()
        data['params'] = utils.loadfile(filenames['params'], 'I')
        data['target'] = utils.loadfile(filenames['target'], 'S')
        if transpose:
            data['target'] = data['target'].T
        if filenames_test == None:
            self.process(data)
        else:
            data_test = dict()
            data_test['params'] = utils.loadfile(filenames_test['params'], 'I')
            data_test['target'] = utils.loadfile(filenames_test['target'], 'S')
            if transpose:
                data_test['target'] = data_test['target'].T
            self.process(data, data_test)
    


    def compute_pod_matrix(self, snapshot_matrix : np.array):
        """ Computes the pod subspace matrix

        Args:
            snapshot_matrix (np.array): contains the matrix to compute the POD
                                        subspace with.
        """
        if self.rpod == False:
            self.subspace, _, _ = svd(
                np.transpose(snapshot_matrix), 
                full_matrices = False
            )
            self.subspace = self.subspace[:,:self.pod_dim]
        else:
            self.subspace, _, _ = randomized_svd(
                np.transpose(snapshot_matrix),
                n_components = self.pod_dim,
                random_state = 0
            )
    


    def pod_projection(self, snapshot_matrix):
        """ Projects the snapshot matrix onto the reduced subspace.

        Args:
            snapshot matrix: the matrix to reduce with POD.
        
        Return:
            The POD coefficients.
        """
        return np.einsum('bi,ij->bj', snapshot_matrix, self.subspace)
    


    def process(self, data, data_test = None):
        """ Backend function which processes the data 
            (splitting, projection, normalization).
        
        Args:
            data: the loaded data. 
            data_test: the test data (if given in a different file).
        """

        n_samples = data['target'].shape[0]
        n_train = int(self.alpha_train * (n_samples / self.N_t)) * self.N_t
        self.n_train = n_train

        # Data splitting
        if data_test == None:
            alpha_test = (1.0 - self.alpha_train) / 2
            n_test = round(alpha_test * (n_samples / self.N_t)) * self.N_t
            self.data['train']['params'] = data['params'][:n_train]
            self.data['val']['params'] = data['params'][n_train:-n_test]
            self.data['test']['params'] = data['params'][-n_test:]
            self.data['train']['target'] = data['target'][:n_train]
            self.data['val']['target'] = data['target'][n_train:-n_test]
            self.data['test']['target'] = data['target'][-n_test:]
        else:
            self.data['train']['params'] = data['params'][:n_train]
            self.data['val']['params'] = data['params'][n_train:]
            self.data['test']['params'] = data_test['params']
            self.data['train']['target'] = data['target'][:n_train]
            self.data['val']['target'] = data['target'][n_train:]
            self.data['test']['target'] = data_test['target']

        # Eventually projects the data onto the reduced subspace
        if self.pod_dim != None:
            self.compute_pod_matrix(self.data['train']['target'])
            self.data['train_fom'] = dict()
            self.data['train_fom']['target'] = np.copy(
                self.data['train']['target'])
            self.data['train']['target'] = self.pod_projection(
                self.data['train']['target'])
            self.data['val']['target'] = self.pod_projection(
                self.data['val']['target'])

        # Normalizes inputs and outputs whenever necessary
        self.normalizer = utils.Normalizer(self.data['train']['params'],
                                           self.data['train']['target'])
        for dataset in ('train', 'val', 'test'):
            self.data[dataset]['params'] = self.normalizer.forward_x(
                self.data[dataset]['params'])
        for dataset in ('train', 'val'):
            self.data[dataset]['target'] = self.normalizer.forward_y(
                self.data[dataset]['target'])

        


    



    
      



