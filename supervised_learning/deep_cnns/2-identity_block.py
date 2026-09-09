#!/usr/bin/env python3
"""ResNet identity block."""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """Build an identity block as described in Deep Residual Learning
    for Image Recognition (2015).

    A_prev: output from the previous layer
    filters: tuple/list (F11, F3, F12)
        F11: number of filters in the first 1x1 convolution
        F3: number of filters in the 3x3 convolution
        F12: number of filters in the second 1x1 convolution
    Returns: the activated output of the identity block
    """
    F11, F3, F12 = filters
    init = K.initializers.he_normal(seed=0)
    conv1 = K.layers.Conv2D(F11, (1, 1), padding='same',
                            kernel_initializer=init)(A_prev)
    bn1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation('relu')(bn1)
    conv2 = K.layers.Conv2D(F3, (3, 3), padding='same',
                            kernel_initializer=init)(act1)
    bn2 = K.layers.BatchNormalization(axis=3)(conv2)
    act2 = K.layers.Activation('relu')(bn2)
    conv3 = K.layers.Conv2D(F12, (1, 1), padding='same',
                            kernel_initializer=init)(act2)
    bn3 = K.layers.BatchNormalization(axis=3)(conv3)
    add = K.layers.Add()([bn3, A_prev])
    output = K.layers.Activation('relu')(add)
    return output
