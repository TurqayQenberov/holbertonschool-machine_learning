#!/usr/bin/env python3
"""
RNN propagation function module
"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN

    Parameters:
        rnn_cell: instance of RNNCell
        X: numpy.ndarray of shape (t, m, i) with input data
        h_0: numpy.ndarray of shape (m, h) with initial hidden state

    Returns:
        H: numpy.ndarray containing all hidden states, shape (t + 1, m, h)
        Y: numpy.ndarray containing all outputs, shape (t, m, o)
    """
    t, m, _ = X.shape
    _, h = h_0.shape

    H = np.zeros((t + 1, m, h))
    H[0] = h_0

    outputs = []
    for step in range(t):
        h_next, y = rnn_cell.forward(H[step], X[step])
        H[step + 1] = h_next
        outputs.append(y)

    Y = np.array(outputs)

    return H, Y
