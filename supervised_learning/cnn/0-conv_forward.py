#!/usr/bin/env python3
"""Convolutional forward propagation."""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Perform forward propagation over a convolutional layer.
    activation: activation function applied to the convolution
    padding: 'same' or 'valid'
    stride: tuple (sh, sw)

    Returns: the output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride
    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph, pw = 0, 0
    A_prev_pad = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    h_out = (h_prev + 2 * ph - kh) // sh + 1
    w_out = (w_prev + 2 * pw - kw) // sw + 1
    Z = np.zeros((m, h_out, w_out, c_new))
    for i in range(h_out):
        for j in range(w_out):
            for k in range(c_new):
                x0 = i * sh
                y0 = j * sw
                slice_a = A_prev_pad[:, x0:x0 + kh, y0:y0 + kw, :]
                Z[:, i, j, k] = np.sum(
                    slice_a * W[:, :, :, k], axis=(1, 2, 3)
                )
    Z = Z + b
    return activation(Z)
