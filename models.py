################################################################################
# Official code implementation of the paper
# "Error estimates for POD-DL-ROMs: a deep learning framework for reduced order
#  modeling of nonlinear parametrized PDEs enhanced by proper orthogonal 
#  decomposition" 
# https://arxiv.org/abs/2305.04680
#
# -> Implementation of neural network models <-
# 
# Authors:     S.Brivio, S.Fresca, N.R.Franco, A.Manzoni 
# Affiliation: MOX Laboratory (Politecnico di Milano, Mathematics Department)
################################################################################

import numpy as np
np.random.seed(1)

import os
import tensorflow as tf
tf.random.set_seed(1)

from utils import set_checkpoint_folder



class Network:
    """
    Neural network model
    """

    def __init__(self, 
                 data_gen, 
                 model_train, 
                 model_test = None, 
                 name = 'net', 
                 save_folder = 'save_dir'):
        self.name = name
        self.save_folder = save_folder
        self.data_gen = data_gen
        self.model_train = model_train
        if model_test == None:
            self.model_test = model_train
        else:
            self.model_test = model_test
        set_checkpoint_folder(self)
    
    
    def summary(self):
        self.model_train.summary()
        self.model_test.summary()

    
    def relative_error_metric(self, y_true, y_pred):
        y_true_unnorm = self.data_gen.normalizer.backward(y_true)
        y_pred_unnorm = self.data_gen.normalizer.backward(y_pred)
        rel_err = tf.reduce_mean(
            tf.reduce_sum((y_true_unnorm - y_pred_unnorm)**2, axis = 1) /
            tf.reduce_sum((y_true_unnorm)**2, axis = 1)
        )
        return rel_err


    def compile(self, lr = 1e-3,
                n_early_stopping = np.inf, 
                n_reduce_lr = np.inf):
    
        # Defines the outer loss for the reconstruction error
        def L_outer(y_true, y_pred):
            return tf.reduce_mean(tf.reduce_sum((y_true - y_pred)**2, axis = 1))
        
        # Implements the save callback
        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True
            )
        # Implements the early stopping callback
        self.es_callback = tf.keras.callbacks.EarlyStopping(
            monitor = 'val_loss', 
            patience = n_early_stopping)
        
        # Implements learning rate decay
        self.lr_decay_callback = tf.keras.callbacks.ReduceLROnPlateau(
            factor = 0.5, 
            patience = n_reduce_lr, 
            monitor = 'val_loss', 
            min_lr = lr/8)
        
        # Compiles the model
        self.model_train.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate = lr),
            loss = [L_outer],
            metrics = [self.relative_error_metric],
            jit_compile = True)
        
        
    def fit(self, batch_size = 1, n_epochs = 1, verbose = 'auto'):

        # Extracts data from data generator
        x_train = self.data_gen.data['train']['params']
        y_train = self.data_gen.data['train']['target']
        x_val = self.data_gen.data['val']['params']
        y_val = self.data_gen.data['val']['target']

        # Trains the model
        if isinstance(self.model_train.input, list):
            if len(self.model_train.input) == 2:
                x_train = x_train, y_train
                x_val = x_val, y_val
        history = self.model_train.fit(
            x_train, 
            y_train, 
            batch_size = batch_size, 
            epochs = n_epochs, 
            validation_data = (x_val, y_val),
            callbacks = [self.model_checkpoint_callback,
                         self.es_callback,
                         self.lr_decay_callback],
            verbose = verbose)
        return history

    
    def load(self):
        self.model_test.load_weights(self.checkpoint_filepath, by_name = 'True')


    def predict(self, x):
        return self.data_gen.normalizer.backward(self.model_test.predict(x))
    
    
    def test(self):
        """
        Tests the trained model computing the relative error
        """
        params_test = self.data_gen.data['test']['params']
        S_test = self.data_gen.data['test']['target']
        S_pred_test = self.predict(params_test)
        PxT = np.prod(
            self.data_gen.normalizer.x_max - self.data_gen.normalizer.x_min)
        E_R = np.sqrt(PxT * np.mean(
            np.linalg.norm(S_pred_test - S_test, axis = 1)**2 /\
            np.linalg.norm(S_test, axis = 1)**2))
        return E_R
    

    
        
        

class PODNetwork(Network):
    """
    (POD + Neural network) model
    """

    def __init__(self, 
                 data_gen, 
                 model_train, 
                 model_test = None, 
                 name = 'podnet', 
                 save_folder = 'save_dir'):
        super(PODNetwork, self).__init__(
            data_gen, model_train, model_test, name, save_folder
        )
        
    
    def relative_error_metric(self, y_true, y_pred):
        y_true_fom = self.lift(self.data_gen.normalizer.backward(y_true))
        y_pred_fom = self.lift(self.data_gen.normalizer.backward(y_pred))
        rel_err = tf.reduce_mean(
            tf.reduce_sum((y_true_fom - y_pred_fom)**2, axis = 1) /
            tf.reduce_sum((y_true_fom)**2, axis = 1)
        )
        return rel_err


    def project(self, x):
        return tf.einsum('bi,ij->bj', x, self.data_gen.subspace)
    

    def lift(self, x):
        return tf.einsum('bj,ij->bi', x, self.data_gen.subspace)
    

    def predict(self, x):
        return self.lift(
            self.data_gen.normalizer.backward(self.model_test.predict(x)))
    

    def test(self):
        """
        Tests the trained model computing the relative error, the upper bound,
        the lower bound and the error decomposition formula
        """

        params_test = self.data_gen.data['test']['params']
        S_train = self.data_gen.data['train_fom']['target']
        S_test = self.data_gen.data['test']['target']
        S_pred_test = self.predict(params_test)

        # Compute quantities of interest
        m = np.min(np.linalg.norm(S_test, axis = 1))
        M = np.max(np.linalg.norm(S_test, axis = 1))
        PxT = np.prod(
            self.data_gen.normalizer.x_max - self.data_gen.normalizer.x_min)
        S_proj_train = S_train @ self.data_gen.subspace @ \
            self.data_gen.subspace.T 
        unexplained_var_train = PxT * np.mean(
            np.linalg.norm(S_proj_train - S_train, 2, axis = 1)**2)
        S_proj_test = S_test @ self.data_gen.subspace @ self.data_gen.subspace.T 
        unexplained_var_test = PxT * np.mean(
            np.linalg.norm(S_proj_test - S_test, 2, axis = 1)**2)
        
        # Sampling error
        E_S = m**(-1) * \
            np.abs(unexplained_var_test - unexplained_var_train)**(1/2)

        # POD error
        E_POD = m**(-1) * np.sqrt(unexplained_var_train)
        E_POD_inf = m**(-1) * np.sqrt(unexplained_var_test)

        # NN error
        E_NN = np.sqrt(PxT * np.mean(
            np.linalg.norm(S_pred_test - S_proj_test, axis = 1)**2 / \
            np.linalg.norm(S_test, axis = 1)**2))
        
        # Relative error
        E_R = np.sqrt(PxT * np.mean(
            np.linalg.norm(S_pred_test - S_test, axis = 1)**2 /\
            np.linalg.norm(S_test, axis = 1)**2))
        
        E_R = E_R.astype('float32')
        E_S = E_S.astype('float32')
        E_POD = E_POD.astype('float32')
        E_POD_inf = E_POD_inf.astype('float32')
        E_NN = E_NN.astype('float32')

	# Print relevant info
        print('\n * Error decomposition * \nE_R = ', E_R,' | E_S = ', E_S,
              ' | E_POD = ', E_POD, ' | E_NN = ', E_NN)
        
        print('\n * Upper and lower bounds * \n', m / M * E_POD_inf,  " <= ",
               E_R, " <= ", E_NN + E_POD + E_S)
        
        # Export predictions
        np.save(
            os.path.join(self.save_folder, 'results' + self.name + '.npy'), 
            S_pred_test
        )

        return {"E_R": E_R, "E_S": E_S, "E_POD": E_POD, \
            "E_NN": E_NN, "upper_bound": E_S + E_POD + E_NN, \
            "lower_bound": m / M * E_POD_inf }

   









