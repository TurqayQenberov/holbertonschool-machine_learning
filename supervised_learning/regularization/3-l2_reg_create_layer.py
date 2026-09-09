#!/usr/bin/env python3
"""Create a layer with L2 regularization"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a neural network layer in tensorFlow that includes
    L2 regularization
    lambtha is the L2 regularization parameter
    Returns: the output of the new layer
    """
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode="fan_avg")
    regularizer = tf.keras.regularizers.L2(lambtha)
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer)
    return layer(prev)
