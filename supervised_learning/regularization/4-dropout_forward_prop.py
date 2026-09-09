#!/usr/bin/env python3
"""
Module containing the dropout_forward_prop function.
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.
    Parameters:
    X is a numpy.ndarray of shape (nx, m) containing the input data
    weights is a dictionary of the weights and of the neural network
    L is the number of layers in the network
    keep_prob is the probability that a node will be kept
    Returns:
    A dictionary containing the outputs of each layer and the dropout mask
    used on each layer.
    """
    cache = {}
    cache['A0'] = X
    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        A_prev = cache['A' + str(i - 1)]
        # Linear step
        Z = np.dot(W, A_prev) + b
        # Activation step
        if i == L:
            # Softmax for the output layer
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            cache['A' + str(i)] = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        else:
            # tanh for hidden layers
            A = np.tanh(Z)
            # Apply inverted dropout
            D = np.random.binomial(1, keep_prob, size=A.shape)
            A = (A * D) / keep_prob
            cache['D' + str(i)] = D
            cache['A' + str(i)] = A
    return cache
