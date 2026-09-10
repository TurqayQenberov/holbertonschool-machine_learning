#!/usr/bin/env python3
"""
This module contains a function that calculates the probability
density function of a Gaussian distribution for a given data set,
mean, and covariance matrix.
"""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian
    distribution

    X is a numpy.ndarray of shape (n, d) containing the data points
        whose PDF should be evaluated
    m is a numpy.ndarray of shape (d,) containing the mean of the
        distribution
    S is a numpy.ndarray of shape (d, d) containing the covariance
        of the distribution

    Returns: P, or None on failure
        P is a numpy.ndarray of shape (n,) containing the PDF values
            for each data point
        All values in P should have a minimum value of 1e-300
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None

    n, d = X.shape
    if m.shape[0] != d or S.shape[0] != d or S.shape[1] != d:
        return None

    det = np.linalg.det(S)
    inv = np.linalg.inv(S)

    diff = X - m

    exponent = -0.5 * np.sum(diff @ inv * diff, axis=1)

    coefficient = 1 / np.sqrt(((2 * np.pi) ** d) * det)

    P = coefficient * np.exp(exponent)

    P = np.maximum(P, 1e-300)

    return P
