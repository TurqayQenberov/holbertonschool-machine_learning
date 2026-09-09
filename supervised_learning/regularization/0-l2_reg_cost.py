#!/usr/bin/env python3
"""
Module containing the l2_reg_cost function.
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.

    Parameters:
    cost (float): The cost of the network without L2 regularization.
    lambtha (float): The regularization parameter.
    weights (dict): A dictionary of the weights and biases (numpy.ndarrays)
                    of the neural network.
    L (int): The number of layers in the neural network.
    m (int): The number of data points used.

    Returns:
    The total cost of the network accounting for L2 regularization.
    """
    l2_sum = 0
    for i in range(1, L + 1):
        l2_sum += np.sum(np.square(weights['W' + str(i)]))

    l2_cost = cost + (lambtha / (2 * m)) * l2_sum
    return l2_cost
