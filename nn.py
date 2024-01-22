################################################################################
# Official code implementation of the paper
# "Error estimates for POD-DL-ROMs: a deep learning framework for reduced order
#  modeling of nonlinear parametrized PDEs enhanced by proper orthogonal 
#  decomposition" 
# https://arxiv.org/abs/2305.04680
#
# -> Basic neural network architectures <-
# 
# Authors:     S.Brivio, S.Fresca, N.R.Franco, A.Manzoni 
# Affiliation: MOX Laboratory (Politecnico di Milano, Mathematics Department)
################################################################################

import numpy as np
np.random.seed(1)

import tensorflow as tf
tf.random.set_seed(1)


class DenseNetwork(tf.keras.Model):
    """
    Implements a dense block of constant width and fixed depth.
    """

    def __init__(self, 
                 width, 
                 depth, 
                 output_dim,
                 activation = tf.keras.layers.LeakyReLU(), 
                 kernel_initializer = 'he_uniform'):
        super(DenseNetwork, self).__init__()
        self.width = width
        self.depth = depth
        self.output_dim = output_dim
        # Defines the first (depth - 1) layers
        self.dense_layers = [
            tf.keras.layers.Dense(self.width,
                                  activation = activation,
                                  kernel_initializer = kernel_initializer)
            for i in range(depth-1)
        ]
        # Defines the last layer
        self.dense_layers.append(
            tf.keras.layers.Dense(output_dim,
                                  activation = 'linear',
                                  kernel_initializer = kernel_initializer)
        )
    
    def call(self, x, training = False):
        for i in range(self.depth):
            x = self.dense_layers[i](x)
        return x



class ResNet(tf.keras.Model):
    def __init__(self, 
                 latent_dim,
                 depth, 
                 output_dim,
                 activation = tf.keras.layers.LeakyReLU(), 
                 kernel_initializer = 'he_uniform'):
        super(ResNet, self).__init__()
        self.latent_dim = latent_dim
        self.depth = depth
        self.output_dim = output_dim
        # Defines the internal resnet layers
        self.w0_layers = [
            tf.keras.layers.Dense(self.latent_dim,
                                  activation = activation,
                                  kernel_initializer = kernel_initializer)
            for i in range(depth)
        ]
        # Defines the external resnet layers
        self.w1_layers = [
            tf.keras.layers.Dense(self.output_dim,
                                  activation = None,
                                  kernel_initializer = kernel_initializer)
            for i in range(depth)
        ]

    def call(self, x, training = False):
        for i in range(self.depth):
            x = self.w0_layers[i](x)
            x = self.w1_layers[i](x)
        return x



def create_dlrom(n_inputs : int, architecture: dict):
    """
    Builds a DL-ROM model from single architectures components
    """

    # Gets the architecture
    reduced_network = architecture.get('reduced_network')
    encoder_network = architecture.get('encoder_network')
    decoder_network = architecture.get('decoder_network')
    
    # Builds the network
    input_reduced = tf.keras.Input((n_inputs), name = 'input_reduced')
    input_encoder = tf.keras.Input((decoder_network.output_dim), 
                                    name = 'input_encoder')
    reduced_network_repr = reduced_network(input_reduced)
    latent_repr = encoder_network(input_encoder)
    output = decoder_network(reduced_network_repr)

    # Creates train and test models
    model_train = tf.keras.models.Model(
            inputs = [input_reduced, input_encoder],
            outputs = [output]
    )
    model_test = tf.keras.models.Model(
                inputs = [input_reduced],
                outputs = [output]
    )
    
    # Adds internal loss to ensure a suitable latent representation
    L_inner = tf.reduce_mean(
        tf.reduce_sum((reduced_network_repr - latent_repr)**2, axis = 1)
    )
    model_train.add_loss(L_inner)

    return model_train, model_test



def create_dnn_from_blocks(n_inputs: int, blocks : list):
    """
    Creates a DNN model from neural network blocks
    """
    
    # Builds the network
    input = tf.keras.Input((n_inputs), name = 'input_reduced')
    output = input
    for i in range(len(blocks)):
        output = blocks[i](output)
    
    # Creates the model
    model = tf.keras.models.Model(
                inputs = [input],
                outputs = [output]
    )

    return model