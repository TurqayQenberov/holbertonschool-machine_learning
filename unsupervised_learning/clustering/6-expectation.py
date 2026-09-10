#!/usr/bin/env python3
"""
This module contains a function that calculates the expectation
step in the Expectation-Maximization algorithm for a Gaussian
Mixture Model.
"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm for a GMM

    X is a numpy.ndarray of shape (n, d) containing the data set
    pi is a numpy.ndarray of shape (k,) containing the priors for
        each cluster
    m is a numpy.ndarray of shape (k, d) containing the centroid
        means for each cluster
    S is a numpy.ndarray of shape (k, d, d) containing the
        covariance matrices for each cluster

    Returns: g, l, or None, None on failure
        g is a numpy.ndarray of shape (k, n) containing the
            posterior probabilities for each data point in each
            cluster
        l is the total log likelihood
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if m.shape[0] != k or m.shape[1] != d:
        return None, None
    if S.shape[0] != k or S.shape[1] != d or S.shape[2] != d:
        return None, None
    if not np.isclose(np.sum(pi), 1):
        return None, None

    g = np.zeros((k, n))

    for i in range(k):
        P = pdf(X, m[i], S[i])
        if P is None:
            return None, None
        g[i] = pi[i] * P

    total = np.sum(g, axis=0)

    g = g / total

    log_likelihood = np.sum(np.log(total))

    return g, log_likelihood
