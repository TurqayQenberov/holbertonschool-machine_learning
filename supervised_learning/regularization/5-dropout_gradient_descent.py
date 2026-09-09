#!/usr/bin/env python3
"""
Module containing the dropout_gradient_descent function.
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a with Dropout regularization
    using gradient descent.
    d dropout masks of each layer
    alpha is the learning rate
    keep_prob is the probability that a node will be kept
    L is the number of layers of the network

    Returns:
    None (updates the weights in place)
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        if i > 1:
            D = cache['D' + str(i - 1)]
            dA = np.dot(W.T, dZ)
            # Apply dropout mask and scale to the gradient
            dA = (dA * D) / keep_prob
            # tanh derivative: g'(z) = 1 - A^2
            dZ = dA * (1 - np.square(A_prev))
        # In-place parameter updates
        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db
