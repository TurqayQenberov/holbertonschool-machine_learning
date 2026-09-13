#!/usr/bin/env python3
"""Creates a Momentum optimizer instance in TensorFlow"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Sets up the gradient descent with momentum optimization
    algorithm in TensorFlow.

    Parameters:
    alpha (float): The learning rate
    beta1 (float): The momentum weight

    Returns:
    tf.keras.optimizers.Optimizer: The configured SGD optimizer with momentum
    """
    return tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
