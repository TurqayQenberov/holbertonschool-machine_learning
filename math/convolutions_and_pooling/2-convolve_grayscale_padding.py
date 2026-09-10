#!/usr/bin/env python3
"""
Convolution with Padding Module
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding.

    Args:
        images: numpy.ndarray with shape (m, h, w)
        kernel: numpy.ndarray with shape (kh, kw)
        padding: tuple of (ph, pw) containing padding height and width

    Returns:
        numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Apply the custom zero-padding to the height and width axes
    images_padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant',
        constant_values=0
    )

    # Calculate output spatial dimensions
    output_h = h + (2 * ph) - kh + 1
    output_w = w + (2 * pw) - kw + 1

    # Initialize the output array
    convolved_images = np.zeros((m, output_h, output_w))

    # Loop over the spatial dimensions of the output image
    for i in range(output_h):
        for j in range(output_w):
            # Slice region of interest from the padded images
            image_patch = images_padded[:, i:i+kh, j:j+kw]

            # Multiply by kernel and sum along spatial axes
            convolved_images[:, i, j] = np.sum(
                image_patch * kernel, axis=(1, 2)
            )

    return convolved_images
