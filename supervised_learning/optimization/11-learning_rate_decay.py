#!/usr/bin/env python3
"""Calculates the learning rate decay using inverse time decay"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using stepwise inverse time decay.

    Parameters:
    alpha (float): The original learning rate
    decay_rate (float): Weight used to determine the rate of decay
    global_step (int): The number of passes of gradient descent elapsed
    decay_step (int): Number of passes before alpha decays further

    Returns:
    float: The updated value for alpha
    """
    # Enforce stepwise behavior by taking the floor division
    step = global_step // decay_step

    # Calculate inverse time decay
    alpha_decayed = alpha / (1 + decay_rate * step)

    return alpha_decayed
