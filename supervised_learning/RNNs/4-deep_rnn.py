#!/usr/bin/env python3
"""
Defines the deep_rnn function for forward propagation in a deep RNN
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN

    Parameters:
        rnn_cells: list of RNNCell instances of length l
        X: numpy.ndarray of shape (t, m, i) with input data
        h_0: numpy.ndarray of shape (l, m, h) with initial
             hidden state

    Returns:
        H: numpy.ndarray containing all hidden states
        Y: numpy.ndarray containing all outputs
    """
    t, m, i = X.shape
    l, _, h = h_0.shape

    # Initialize H with shape (t + 1, l, m, h)
    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0

    # Initialize Y with shape (t, m, o)
    o = rnn_cells[-1].by.shape[1]
    Y = np.zeros((t, m, o))

    # Iterate through all time steps
    for step in range(t):
        x_t = X[step]

        # Iterate through each layer in the deep RNN
        for layer in range(l):
            cell = rnn_cells[layer]
            h_prev = H[step, layer]

            # Forward propagation for current cell/layer
            h_next, y_next = cell.forward(h_prev, x_t)

            # Update hidden state for current step and layer
            H[step + 1, layer] = h_next

            # Output of current layer becomes input to the next layer
            x_t = h_next

        # Save output of the last layer
        Y[step] = y_next

    return H, Y
