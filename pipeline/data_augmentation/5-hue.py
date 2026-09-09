#!/usr/bin/env python3
"""Image hue augmentation."""
import tensorflow as tf


def change_hue(image, delta):
    """Change the hue of an image.

    image: 3D tf.Tensor containing the image to change
    delta: the amount the hue should change
    Returns: the altered image
    """
    return tf.image.adjust_hue(image, delta)
