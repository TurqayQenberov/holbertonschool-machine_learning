#!/usr/bin/env python3
"""Creates a learning rate decay schedule in TensorFlow"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates a learning rate decay operation in TensorFlow using
    inverse time decay in a stepwise fashion.

    Parameters:
    alpha (float): The original learning rate
    decay_rate (float): The rate at which alpha will decay
    decay_step (int): Number of passes before alpha decays further

    Returns:
    tf.keras.optimizers.schedules.LearningRateSchedule: The decay schedule
    """
    class StepwiseInverseTimeDecay(
        tf.keras.optimizers.schedules.LearningRateSchedule
    ):
        """Custom learning rate schedule for stepwise inverse time decay"""
        def __init__(self, alpha, decay_rate, decay_step):
            self.alpha = alpha
            self.decay_rate = decay_rate
            self.decay_step = decay_step

        def __call__(self, step):
            # Cast the step to float32 for safe mathematical operations
            step = tf.cast(step, tf.float32)
            # Implement the floor/staircase division step calculation
            decay_steps_done = tf.floor(step / self.decay_step)
            # Apply the inverse time decay formula
            return self.alpha / (1.0 + self.decay_rate * decay_steps_done)

    return StepwiseInverseTimeDecay(alpha, decay_rate, decay_step)
