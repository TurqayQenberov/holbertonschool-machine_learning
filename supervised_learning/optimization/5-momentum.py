#!/usr/bin/env python3
"""Updates a variable using Gradient Descent with Momentum"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using the gradient descent with momentum algorithm.

    Parameters:
    alpha (float): The learning rate
    beta1 (float): The momentum weight
    var (numpy.ndarray): The variable to be updated
    grad (numpy.ndarray): The gradient of var
    v (numpy.ndarray): The previous first moment of var

    Returns:
    tuple: (updated_var, new_v)
    """
    # Calculate the new velocity (first moment)
    v_new = beta1 * v + (1 - beta1) * grad

    # Update the variable weights
    var_new = var - alpha * v_new

    return var_new, v_new
