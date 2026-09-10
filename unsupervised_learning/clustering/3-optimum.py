#!/usr/bin/env python3
"""Optimize k"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance

    X is a numpy.ndarray of shape (n, d) containing the data set
    kmin is a positive integer containing the minimum number of
        clusters to check for (inclusive)
    kmax is a positive integer containing the maximum number of
        clusters to check for (inclusive)
    iterations is a positive integer containing the maximum number
        of iterations for K-means

    Returns: results, d_vars, or None, None on failure
        results is a list containing the outputs of K-means for
            each cluster size
        d_vars is a list containing the difference in variance from
            the smallest cluster size for each cluster size
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if not isinstance(kmax, int) or kmax <= 0:
        return None, None
    if kmin >= kmax:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    results = []
    d_vars = []
    base_var = None

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None or clss is None:
            return None, None

        results.append((C, clss))

        var = variance(X, C)
        if var is None:
            return None, None

        if k == kmin:
            base_var = var

        d_vars.append(base_var - var)

    return results, d_vars
