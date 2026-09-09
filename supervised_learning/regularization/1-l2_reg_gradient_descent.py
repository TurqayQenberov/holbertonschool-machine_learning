#!/usr/bin/env python3
"""Gradient descent with L2 regularization"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using
    gradient descent with L2 regularization
    Y is a one-hot numpy.ndarray of shape (classes, m) that
    contains the correct labels for the data
        classes is the number of classes
        m is the number of data points
    except the last, which uses a softmax activation
    The weights and biases of the network are updated in place
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y
    for layer in range(L, 0, -1):
        A_prev = cache['A' + str(layer - 1)]
        W = weights['W' + str(layer)]
        b = weights['b' + str(layer)]
        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        if layer > 1:
            dZ = np.matmul(W.T, dZ) * (1 - A_prev ** 2)
        weights['W' + str(layer)] = W - alpha * dW
        weights['b' + str(layer)] = b - alpha * db
