#!/usr/bin/env python3
"""Image contrast augmentation."""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """Randomly adjust the contrast of an image.

    image: 3D tf.Tensor representing the input image
    lower: float, lower bound of the random contrast factor range
    upper: float, upper bound of the random contrast factor range
    Returns: the contrast-adjusted image
    """
    return tf.image.random_contrast(image, lower, upper)
