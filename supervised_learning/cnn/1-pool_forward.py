#!/usr/bin/env python3
"""Pooling forward propagation."""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Perform forward propagation over a pooling layer.

    A_prev: ndarray (m, h_prev, w_prev, c_prev), previous layer output
    kernel_shape: tuple (kh, kw), pooling kernel size
    stride: tuple (sh, sw)
    mode: 'max' or 'avg'
    Returns: the output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    h_out = (h_prev - kh) // sh + 1
    w_out = (w_prev - kw) // sw + 1
    A = np.zeros((m, h_out, w_out, c_prev))
    for i in range(h_out):
        for j in range(w_out):
            x0 = i * sh
            y0 = j * sw
            slice_a = A_prev[:, x0:x0 + kh, y0:y0 + kw, :]
            if mode == 'max':
                A[:, i, j, :] = np.max(slice_a, axis=(1, 2))
            else:
                A[:, i, j, :] = np.mean(slice_a, axis=(1, 2))
    return A
