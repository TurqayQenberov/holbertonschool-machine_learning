#!/usr/bin/env python3
"""
Same Convolution Module
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.

    Args:
        images: numpy.ndarray with shape (m, h, w)
        kernel: numpy.ndarray with shape (kh, kw)

    Returns:
        numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate padding to ensure output size matches input size (h, w)
    # Using integer division to handle odd and even kernels properly
    if kh % 2 == 0:
        ph = kh // 2
    else:
        ph = (kh - 1) // 2

    if kw % 2 == 0:
        pw = kw // 2
    else:
        pw = (kw - 1) // 2

    # Pad only the height (axis 1) and width (axis 2)
    # The batch size axis (axis 0) is not padded
    images_padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant',
        constant_values=0
    )

    # Initialize output array with the same spatial dimensions as input
    convolved_images = np.zeros((m, h, w))

    # Loop over the spatial dimensions of the output image
    for i in range(h):
        for j in range(w):
            # Slice the current region of interest from the padded images
            image_patch = images_padded[:, i:i+kh, j:j+kw]

            # Multiply patch by the kernel and sum across spatial axes
            convolved_images[:, i, j] = np.sum(
                image_patch * kernel, axis=(1, 2)
            )

    return convolved_images
