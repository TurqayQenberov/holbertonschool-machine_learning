#!/usr/bin/env python3
"""Creates a batch normalization layer in TensorFlow"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in TensorFlow.

    Parameters:
    prev (tf.Tensor): The activated output of the previous layer
    n (int): The number of nodes in the layer to be created
    activation (callable): The activation function to be applied

    Returns:
    tf.Tensor: A tensor representing the activated output for the layer
    """
    # 1. Base Dense layer with variance scaling initializer
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense_layer = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=init,
        use_bias=False
    )
    Z = dense_layer(prev)

    # 2. Batch Normalization layer with explicit parameters
    batch_norm = tf.keras.layers.BatchNormalization(
        beta_initializer='zeros',
        gamma_initializer='ones',
        epsilon=1e-7
    )
    # CRITICAL: Pass training=True to force batch statistic calculation
    Z_norm = batch_norm(Z, training=True)

    # 3. Apply activation function
    if activation is not None:
        return activation(Z_norm)

    return Z_norm
