#!/usr/bin/env python3
"""
Valid Convolution Module
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

    Args:
        images: numpy.ndarray with shape (m, h, w)
        kernel: numpy.ndarray with shape (kh, kw)

    Returns:
        numpy.ndarray containing the convolved images.
    """
    # Extract dimensions from the inputs
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate the dimensions of the output image
    output_h = h - kh + 1
    output_w = w - kw + 1

    # Initialize an array of zeros to store the convolved images
    convolved_images = np.zeros((m, output_h, output_w))

    # Loop over the spatial dimensions of the output image
    for i in range(output_h):
        for j in range(output_w):
            # Slice region of interest (patch) across all images
            image_patch = images[:, i:i+kh, j:j+kw]

            # Multiply patch by kernel and sum over spatial axes
            convolved_images[:, i, j] = np.sum(
                image_patch * kernel, axis=(1, 2)
            )

    return convolved_images
