#!/usr/bin/env python3
"""
Convolution with Channels Module
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with channels.

    Args:
        images: numpy.ndarray with shape (m, h, w, c)
        kernel: numpy.ndarray with shape (kh, kw, c)
        padding: tuple of (ph, pw), 'same', or 'valid'
        stride: tuple of (sh, sw) containing stride height and width

    Returns:
        numpy.ndarray containing the convolved images.
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    # Determine padding height (ph) and width (pw)
    if padding == 'same':
        ph = (((h - 1) * sh + kh - h) + 1) // 2
        pw = (((w - 1) * sw + kw - w) + 1) // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    elif isinstance(padding, tuple):
        ph, pw = padding

    # Apply zero-padding only to spatial dimensions (height and width)
    # Axes: 0 (batch), 1 (height), 2 (width), 3 (channels)
    images_padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    # Calculate output dimensions
    output_h = ((h + (2 * ph) - kh) // sh) + 1
    output_w = ((w + (2 * pw) - kw) // sw) + 1

    # Initialize the output array (output is 2D grayscale per image)
    convolved_images = np.zeros((m, output_h, output_w))

    # Loop over the spatial dimensions of the output image
    for i in range(output_h):
        for j in range(output_w):
            # Calculate top-left starting indices
            h_start = i * sh
            w_start = j * sw

            # Slice current 3D patch across all images: shape (m, kh, kw, c)
            image_patch = images_padded[
                :,
                h_start: h_start + kh,
                w_start: w_start + kw,
                :
            ]

            # Multiply patch by kernel and sum across axes 1, 2, and 3
            convolved_images[:, i, j] = np.sum(
                image_patch * kernel, axis=(1, 2, 3)
            )

    return convolved_images
