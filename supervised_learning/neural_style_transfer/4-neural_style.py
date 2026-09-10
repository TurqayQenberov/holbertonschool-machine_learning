#!/usr/bin/env python3
"""Defines the NST class that performs tasks for neural style transfer"""

import numpy as np
import tensorflow as tf


class NST:
    """Performs tasks for Neural Style Transfer"""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor

        parameters:
            style_image [numpy.ndarray of shape (h, w, 3)]:
                image used as style reference
            content_image [numpy.ndarray of shape (h, w, 3)]:
                image used as content reference
            alpha [float]: weight for content cost
            beta [float]: weight for style cost
        """
        if not isinstance(style_image, np.ndarray) or \
                len(style_image.shape) != 3 or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(content_image, np.ndarray) or \
                len(content_image.shape) != 3 or content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if (not isinstance(alpha, (int, float))) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if (not isinstance(beta, (int, float))) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixel values are between 0 and 1
        and its largest side is 512 pixels

        parameters:
            image [numpy.ndarray of shape (h, w, 3)]:
                image to be scaled

        returns:
            the scaled image as a tf.tensor of shape (1, h_new, w_new, 3)
            where max(h_new, w_new) == 512 and min(h_new, w_new) is scaled
            proportionately
        """
        if not isinstance(image, np.ndarray) or \
                len(image.shape) != 3 or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape
        if h > w:
            h_new = 512
            w_new = int(w * (512 / h))
        else:
            w_new = 512
            h_new = int(h * (512 / w))

        image = image[tf.newaxis, :]
        image = tf.image.resize(
            image, size=(h_new, w_new), method='bicubic')

        image = image / 255
        image = tf.clip_by_value(image, 0, 1)

        return image

    def load_model(self):
        """
        Creates the model used to calculate cost from the VGG19 Keras
        base model, replacing MaxPooling2D layers with AveragePooling2D
        """
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )
        vgg.save("vgg_base_model")
        custom_objects = {
            'MaxPooling2D': tf.keras.layers.AveragePooling2D
        }
        vgg = tf.keras.models.load_model(
            "vgg_base_model",
            custom_objects=custom_objects
        )

        style_outputs = []
        content_output = None

        for layer in vgg.layers:
            if layer.name in self.style_layers:
                style_outputs.append(layer.output)
            if layer.name == self.content_layer:
                content_output = layer.output
            layer.trainable = False

        outputs = style_outputs + [content_output]

        self.model = tf.keras.models.Model(vgg.input, outputs)
        return self.model

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates the gram matrix of a layer

        parameters:
            input_layer [tf.Tensor or tf.Variable of shape (1, h, w, c)]:
                contains the layer output whose gram matrix should be
                calculated

        returns:
            a tf.Tensor of shape (1, c, c) containing the gram matrix
            of input_layer
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or \
                len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        _, h, w, c = input_layer.shape

        F = tf.reshape(input_layer, (h * w, c))
        n = tf.shape(F)[0]

        gram = tf.matmul(F, F, transpose_a=True)
        gram = tf.expand_dims(gram, axis=0)
        gram /= tf.cast(n, tf.float32)

        return gram

    def generate_features(self):
        """
        Extracts the features used to calculate neural style cost

        Sets the public instance attributes:
            gram_style_features - a list of gram matrices calculated
                from the style layer outputs of the style image
            content_feature - the content layer output of the
                content image
        """
        preprocess = tf.keras.applications.vgg19.preprocess_input

        style_image = preprocess(self.style_image * 255)
        content_image = preprocess(self.content_image * 255)

        style_features = self.model(style_image)[:-1]
        content_feature = self.model(content_image)[-1]

        self.gram_style_features = [
            self.gram_matrix(style_output) for style_output in style_features
        ]
        self.content_feature = content_feature

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculates the style cost for a single layer

        parameters:
            style_output [tf.Tensor of shape (1, h, w, c)]:
                contains the layer style output of the generated image
            gram_target [tf.Tensor of shape (1, c, c)]:
                the gram matrix of the target style output for
                that layer

        returns:
            the layer's style cost
        """
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) or \
                len(style_output.shape) != 4:
            raise TypeError("style_output must be a tensor of rank 4")

        c = style_output.shape[-1]
        gram_valid = (
            isinstance(gram_target, (tf.Tensor, tf.Variable)) and
            len(gram_target.shape) == 3 and
            gram_target.shape[0] == 1 and
            gram_target.shape[1] == c and
            gram_target.shape[2] == c
        )
        if not gram_valid:
            raise TypeError(
                "gram_target must be a tensor of shape [1, {}, {}]".format(
                    c, c))

        gram_style = self.gram_matrix(style_output)

        return tf.reduce_mean(tf.square(gram_style - gram_target))
