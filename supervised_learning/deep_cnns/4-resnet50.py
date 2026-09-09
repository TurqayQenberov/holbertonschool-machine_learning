#!/usr/bin/env python3
"""ResNet-50 architecture."""
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Build the ResNet-50 architecture as described in Deep Residual
    Learning for Image Recognition (2015).

    Input data shape is assumed to be (224, 224, 3).
    Returns: the keras model
    """
    init = K.initializers.he_normal(seed=0)
    X = K.Input(shape=(224, 224, 3))
    conv1 = K.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same',
                            kernel_initializer=init)(X)
    bn1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation('relu')(bn1)
    pool1 = K.layers.MaxPooling2D((3, 3), strides=(2, 2),
                                  padding='same')(act1)
    conv2 = projection_block(pool1, [64, 64, 256], s=1)
    conv2 = identity_block(conv2, [64, 64, 256])
    conv2 = identity_block(conv2, [64, 64, 256])
    conv3 = projection_block(conv2, [128, 128, 512], s=2)
    conv3 = identity_block(conv3, [128, 128, 512])
    conv3 = identity_block(conv3, [128, 128, 512])
    conv3 = identity_block(conv3, [128, 128, 512])
    conv4 = projection_block(conv3, [256, 256, 1024], s=2)
    conv4 = identity_block(conv4, [256, 256, 1024])
    conv4 = identity_block(conv4, [256, 256, 1024])
    conv4 = identity_block(conv4, [256, 256, 1024])
    conv4 = identity_block(conv4, [256, 256, 1024])
    conv4 = identity_block(conv4, [256, 256, 1024])
    conv5 = projection_block(conv4, [512, 512, 2048], s=2)
    conv5 = identity_block(conv5, [512, 512, 2048])
    conv5 = identity_block(conv5, [512, 512, 2048])
    pool2 = K.layers.AveragePooling2D((7, 7), padding='same')(conv5)
    output = K.layers.Dense(1000, activation='softmax',
                            kernel_initializer=init)(pool2)
    model = K.models.Model(inputs=X, outputs=output)
    return model
