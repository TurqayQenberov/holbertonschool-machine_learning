#!/usr/bin/env python3
"""
Multiple Kernels Convolution Module
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels.

    Args:
        images: numpy.ndarray with shape (m, h, w, c)
        kernels: numpy.ndarray with shape (kh, kw, c, nc)
        padding: tuple of (ph, pw), 'same', or 'valid'
        stride: tuple of (sh, sw) containing stride height and width

    Returns:
        numpy.ndarray containing the convolved images.
    """
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
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
    images_padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    # Calculate output dimensions
    output_h = ((h + (2 * ph) - kh) // sh) + 1
    output_w = ((w + (2 * pw) - kw) // sw) + 1

    # Initialize the output array with the third spatial axis and nc channels
    convolved_images = np.zeros((m, output_h, output_w, nc))

    # Loop over the spatial dimensions of the output image and the kernels
    for i in range(output_h):
        for j in range(output_w):
            for k in range(nc):
                # Calculate starting indices
                h_start = i * sh
                w_start = j * sw

                # Slice current patch across all images
                image_patch = images_padded[
                    :,
                    h_start: h_start + kh,
                    w_start: w_start + kw,
                    :
                ]

                # Select the specific kernel k
                current_kernel = kernels[:, :, :, k]

                # Multiply patch by the kernel and sum across axes 1, 2, and 3
                convolved_images[:, i, j, k] = np.sum(
                    image_patch * current_kernel, axis=(1, 2, 3)
                )

    return convolved_images
