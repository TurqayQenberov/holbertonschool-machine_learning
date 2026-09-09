#!/usr/bin/env python3
"""Normalizes an unactivated output using Batch Normalization"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network using
    batch normalization.

    Parameters:
    Z (numpy.ndarray): Matrix of shape (m, n) to be normalized
    gamma (numpy.ndarray): Scale parameters for batch normalization
    beta (numpy.ndarray): Offset parameters for batch normalization
    epsilon (float): Small number to avoid division by zero

    Returns:
    numpy.ndarray: The normalized Z matrix
    """
    # Calculate mini-batch mean and variance down the columns (axis=0)
    mean = np.mean(Z, axis=0)
    # Using explicit population variance matching standard batch norm math
    variance = np.var(Z, axis=0)

    # Standardize Z
    Z_hat = (Z - mean) / np.sqrt(variance + epsilon)

    # Scale and shift
    Z_norm = gamma * Z_hat + beta

    return Z_norm
