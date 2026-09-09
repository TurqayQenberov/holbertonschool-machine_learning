#!/usr/bin/env python3
"""Random image brightness adjustment."""
import tensorflow as tf


def change_brightness(image, max_delta):
    """Randomly changes the brightness of an image.

    image: a 3D tf.Tensor containing the image to change
    max_delta: the maximum amount the image should be brightened
        (or darkened)

    Returns: the altered image
    """
    return tf.image.random_brightness(image, max_delta)
