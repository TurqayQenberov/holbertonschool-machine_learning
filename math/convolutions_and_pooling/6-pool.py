#!/usr/bin/env python3
"""
Pooling Module
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.

    Args:
        images: numpy.ndarray with shape (m, h, w, c)
        kernel_shape: tuple of (kh, kw) containing the pooling kernel shape
        stride: tuple of (sh, sw) containing height and width strides
        mode: string indicating pooling type, either 'max' or 'avg'

    Returns:
        numpy.ndarray containing the pooled images.
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate output dimensions for pooling (standard valid-style division)
    output_h = ((h - kh) // sh) + 1
    output_w = ((w - kw) // sw) + 1

    # Initialize the output array with pooled spatial dimensions and channels
    pooled_images = np.zeros((m, output_h, output_w, c))

    # Loop over the spatial dimensions of the output image
    for i in range(output_h):
        for j in range(output_w):
            # Calculate top-left starting indices
            h_start = i * sh
            w_start = j * sw

            # Slice the current patch across all images: shape (m, kh, kw, c)
            image_patch = images[
                :,
                h_start: h_start + kh,
                w_start: w_start + kw,
                :
            ]

            # Apply the appropriate aggregation function across spatial axes
            if mode == 'max':
                pooled_images[:, i, j, :] = np.max(image_patch, axis=(1, 2))
            elif mode == 'avg':
                pooled_images[:, i, j, :] = np.mean(image_patch, axis=(1, 2))

    return pooled_images
