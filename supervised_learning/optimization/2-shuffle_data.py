#!/usr/bin/env python3
"""Shuffles data points in two matrices synchronously"""
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way.

    Parameters:
    X (numpy.ndarray): Matrix of shape (m, nx) to shuffle
    Y (numpy.ndarray): Matrix of shape (m, ny) to shuffle

    Returns:
    tuple: (X_shuffled, Y_shuffled)
    """
    # Get a permutation of indices based on the number of data points (m)
    permutation = np.random.permutation(X.shape[0])

    # Use advanced indexing to shuffle both arrays identically
    return X[permutation], Y[permutation]
