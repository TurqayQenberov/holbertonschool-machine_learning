#!/usr/bin/env python3
"""Random image cropping."""
import tensorflow as tf


def crop_image(image, size):
    """Performs a random crop of an image.

    image: a 3D tf.Tensor containing the image to crop
    size: a tuple containing the size of the crop

    Returns: the cropped image
    """
    return tf.image.random_crop(image, size)
