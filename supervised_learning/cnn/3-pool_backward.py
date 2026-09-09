#!/usr/bin/env python3
"""Pooling back propagation."""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Perform back propagation over a pooling layer.

    dA: ndarray (m, h_new, w_new, c), gradients of the output
    A_prev: ndarray (m, h_prev, w_prev, c), previous layer output
    kernel_shape: tuple (kh, kw), pooling kernel size
    stride: tuple (sh, sw)
    mode: 'max' or 'avg'
    Returns: the partial derivatives with respect to A_prev
    """
    m, h_new, w_new, c = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride
    dA_prev = np.zeros_like(A_prev)
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for ch in range(c):
                    x0 = h * sh
                    y0 = w * sw
                    if mode == 'max':
                        slice_a = A_prev[i, x0:x0 + kh, y0:y0 + kw, ch]
                        mask = slice_a == np.max(slice_a)
                        dA_prev[i, x0:x0 + kh, y0:y0 + kw, ch] += (
                            mask * dA[i, h, w, ch])
                    else:
                        avg = dA[i, h, w, ch] / (kh * kw)
                        dA_prev[i, x0:x0 + kh, y0:y0 + kw, ch] += (
                            np.ones((kh, kw)) * avg)
    return dA_prev
