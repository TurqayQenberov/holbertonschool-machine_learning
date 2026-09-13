#!/usr/bin/env python3
"""
Defines the LSTMCell class that represents an LSTM unit
"""
import numpy as np


class LSTMCell:
    """
    Represents an LSTM unit
    """

    def __init__(self, i, h, o):
        """
        Class constructor

        Parameters:
            i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        """
        # Weights initialized using a normal distribution
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))

        # Biases initialized as zeros
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step

        Parameters:
            h_prev: numpy.ndarray of shape (m, h)
                    containing previous hidden state
            c_prev: numpy.ndarray of shape (m, h)
                    containing previous cell state
            x_t: numpy.ndarray of shape (m, i)
                 containing input data for the cell

        Returns:
            h_next: next hidden state
            c_next: next cell state
            y: output of the cell (with softmax activation)
        """
        # Concatenate previous hidden state and input data: shape (m, i + h)
        x_concat = np.concatenate((h_prev, x_t), axis=1)

        # Gate activations
        f_t = 1 / (1 + np.exp(
            -(np.matmul(x_concat, self.Wf) + self.bf)
        ))
        u_t = 1 / (1 + np.exp(
            -(np.matmul(x_concat, self.Wu) + self.bu)
        ))
        c_tilde = np.tanh(
            np.matmul(x_concat, self.Wc) + self.bc
        )

        # Cell state update
        c_next = f_t * c_prev + u_t * c_tilde

        # Output gate and hidden state update
        o_t = 1 / (1 + np.exp(
            -(np.matmul(x_concat, self.Wo) + self.bo)
        ))
        h_next = o_t * np.tanh(c_next)

        # Cell output with softmax activation
        y_linear = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y_linear) / np.sum(np.exp(y_linear), axis=1, keepdims=True)

        return h_next, c_next, y
