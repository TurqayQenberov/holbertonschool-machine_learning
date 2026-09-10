#!/usr/bin/env python3
"""
Strided Convolution Module
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images with custom padding and stride.

    Args:
        images: numpy.ndarray with shape (m, h, w)
        kernel: numpy.ndarray with shape (kh, kw)
        padding: tuple of (ph, pw), 'same', or 'valid'
        stride: tuple of (sh, sw) containing stride height and width

    Returns:
        numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Determine padding height (ph) and width (pw) based on padding argument
    if padding == 'same':
        # Formula for same padding with stride:
        # ph = ceil(((h - 1) * sh + kh - h) / 2)
        ph = (((h - 1) * sh + kh - h) + 1) // 2
        pw = (((w - 1) * sw + kw - w) + 1) // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    elif isinstance(padding, tuple):
        ph, pw = padding

    # Apply zero-padding
    images_padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant',
        constant_values=0
    )

    # Calculate output dimensions
    output_h = ((h + (2 * ph) - kh) // sh) + 1
    output_w = ((w + (2 * pw) - kw) // sw) + 1

    # Initialize the output array
    convolved_images = np.zeros((m, output_h, output_w))

    # Loop over the spatial dimensions of the output image
    for i in range(output_h):
        for j in range(output_w):
            # Calculate the top-left index of the patch in padded image
            h_start = i * sh
            w_start = j * sw

            # Slice the current patch across all images
            image_patch = images_padded[
                :,
                h_start: h_start + kh,
                w_start: w_start + kw
            ]

            # Multiply patch by the kernel and sum across spatial axes
            convolved_images[:, i, j] = np.sum(
                image_patch * kernel, axis=(1, 2)
            )

    return convolved_images
