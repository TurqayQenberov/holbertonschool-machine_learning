#!/usr/bin/env python3
"""GRUCell sinifi"""

import numpy as np


class GRUCell:
    """Gated Recurrent Unit (GRU) hucresini temsil eden sinif"""

    def __init__(self, i, h, o):
        """Initialize GRUCell"""
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))

        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Forward propagation for one time step"""
        concat_input = np.concatenate((h_prev, x_t), axis=1)

        z = 1 / (1 + np.exp(
            -(np.matmul(concat_input, self.Wz) + self.bz)
        ))

        r = 1 / (1 + np.exp(
            -(np.matmul(concat_input, self.Wr) + self.br)
        ))

        concat_reset = np.concatenate((r * h_prev, x_t), axis=1)

        h_tilde = np.tanh(
            np.matmul(concat_reset, self.Wh) + self.bh
        )

        h_next = (1 - z) * h_prev + z * h_tilde

        y_linear = np.matmul(h_next, self.Wy) + self.by

        y = np.exp(y_linear) / np.sum(
            np.exp(y_linear),
            axis=1,
            keepdims=True
        )

        return h_next, y
