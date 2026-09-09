#!/usr/bin/env python3
"""Image flip augmentation."""
import tensorflow as tf


def flip_image(image):
    """Flip an image horizontally.

    image: 3D tf.Tensor containing the image to flip
    Returns: the flipped image
    """
    return tf.image.flip_left_right(image)
