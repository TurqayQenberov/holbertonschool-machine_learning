#!/usr/bin/env python3
"""
This module contains a function that calculates the maximization
step in the Expectation-Maximization algorithm for a Gaussian
Mixture Model.
"""
import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM

    X is a numpy.ndarray of shape (n, d) containing the data set
    g is a numpy.ndarray of shape (k, n) containing the posterior
        probabilities for each data point in each cluster

    Returns: pi, m, S, or None, None, None on failure
        pi is a numpy.ndarray of shape (k,) containing the updated
            priors for each cluster
        m is a numpy.ndarray of shape (k, d) containing the updated
            centroid means for each cluster
        S is a numpy.ndarray of shape (k, d, d) containing the
            updated covariance matrices for each cluster
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    n, d = X.shape
    k, n_g = g.shape

    if n != n_g:
        return None, None, None

    sums = np.sum(g, axis=0)
    if not np.all(np.isclose(sums, 1)):
        return None, None, None

    Nk = np.sum(g, axis=1)

    pi = Nk / n

    m = np.matmul(g, X) / Nk[:, np.newaxis]

    S = np.zeros((k, d, d))

    for i in range(k):
        diff = X - m[i]
        S[i] = np.matmul((g[i, :, np.newaxis] * diff).T, diff) / Nk[i]

    return pi, m, S
