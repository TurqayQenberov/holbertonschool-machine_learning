#!/usr/bin/env python3
"""Convolutional back propagation."""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Performs back propagation over a convolutional layer of a NN.

    dZ: (m, h_new, w_new, c_new) partial derivs w.r.t. unactivated
        output of the conv layer
    A_prev: (m, h_prev, w_prev, c_prev) output of the previous layer
    W: (kh, kw, c_prev, c_new) kernels for the convolution
    b: (1, 1, 1, c_new) biases applied to the convolution
    padding: string 'same' or 'valid'
    stride: tuple (sh, sw) strides for the convolution

    Returns: dA_prev, dW, db
    """
    m, h_new, w_new, c_new = dZ.shape
    _, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride
    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph, pw = 0, 0
    A_prev_pad = np.pad(
        A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode="constant"
    )
    dA_prev_pad = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(h_new):
            vs = h * sh
            ve = vs + kh
            for w in range(w_new):
                hs = w * sw
                he = hs + kw
                for c in range(c_new):
                    a_slice = a_prev_pad[vs:ve, hs:he, :]
                    da_prev_pad[vs:ve, hs:he, :] += (
                        W[:, :, :, c] * dZ[i, h, w, c]
                    )
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
    if padding == "same":
        dA_prev = dA_prev_pad[:, ph:ph + h_prev, pw:pw + w_prev, :]
    else:
        dA_prev = dA_prev_pad
    return dA_prev, dW, db
