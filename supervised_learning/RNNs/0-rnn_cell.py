#!/usr/bin/env python3
"""
RNNCell Module
"""
import numpy as np


class RNNCell:
    """
    Represents a cell of a simple RNN
    """

    def __init__(self, i, h, o):
        """
        Class constructor for RNNCell

        Parameters:
            i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        """
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step

        Parameters:
            h_prev: numpy.ndarray of shape (m, h) with previous hidden state
            x_t: numpy.ndarray of shape (m, i) with data input for the cell

        Returns:
            h_next: next hidden state
            y: output of the cell
        """
        concat_input = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concat_input, self.Wh) + self.bh)

        y_linear = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y_linear) / np.sum(np.exp(y_linear), axis=1, keepdims=True)

        return h_next, y
