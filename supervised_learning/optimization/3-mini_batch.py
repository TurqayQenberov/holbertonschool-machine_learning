#!/usr/bin/env python3
"""Creates mini-batches for training from a dataset"""
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches from a dataset for mini-batch gradient descent.

    Parameters:
    X (numpy.ndarray): Matrix of shape (m, nx) representing input data
    Y (numpy.ndarray): Matrix of shape (m, ny) representing labels
    batch_size (int): Number of data points in a batch

    Returns:
    list: List of mini-batches containing tuples (X_batch, Y_batch)
    """
    # First shuffle X and Y synchronously
    X_shuffled, Y_shuffled = shuffle_data(X, Y)

    m = X.shape[0]
    mini_batches = []

    # Loop through data to extract full batches
    for i in range(0, m, batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        Y_batch = Y_shuffled[i:i + batch_size]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
