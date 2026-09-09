#!/usr/bin/env python3
"""Image rotation augmentation."""
import tensorflow as tf


def rotate_image(image):
    """Rotate an image by 90 degrees counter-clockwise.

    image: 3D tf.Tensor containing the image to rotate
    Returns: the rotated image
    """
    return tf.image.rot90(image)
