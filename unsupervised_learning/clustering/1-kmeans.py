#!/usr/bin/env python3
"""K-means"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means

    X is a numpy.ndarray of shape (n, d) containing the dataset
    k is a positive integer containing the number of clusters

    Returns: a numpy.ndarray of shape (k, d) containing the initialized
             centroids for each cluster, or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    low = X.min(axis=0)
    high = X.max(axis=0)

    centroids = np.random.uniform(low, high, size=(k, X.shape[1]))

    return centroids


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset

    X is a numpy.ndarray of shape (n, d) containing the dataset
    k is a positive integer containing the number of clusters
    iterations is a positive integer containing the maximum number
        of iterations that should be performed

    Returns: C, clss, or None, None on failure
        C is a numpy.ndarray of shape (k, d) containing the centroid
            means for each cluster
        clss is a numpy.ndarray of shape (n,) containing the index of
            the cluster in C that each data point belongs to
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    C = initialize(X, k)
    if C is None:
        return None, None

    low = X.min(axis=0)
    high = X.max(axis=0)

    for i in range(iterations):
        C_prev = C.copy()

        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        for j in range(k):
            if np.sum(clss == j) == 0:
                C[j] = np.random.uniform(low, high, size=(1, d))
            else:
                C[j] = X[clss == j].mean(axis=0)

        if np.array_equal(C, C_prev):
            distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
            clss = np.argmin(distances, axis=1)
            return C, clss

    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    clss = np.argmin(distances, axis=1)
    return C, clss
