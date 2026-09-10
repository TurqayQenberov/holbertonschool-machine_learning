#!/usr/bin/env python3
"""
This module contains a function that finds the best number of
clusters for a Gaussian Mixture Model using the Bayesian
Information Criterion.
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using the Bayesian
    Information Criterion

    X is a numpy.ndarray of shape (n, d) containing the data set
    kmin is a positive integer containing the minimum number of
        clusters to check for (inclusive)
    kmax is a positive integer containing the maximum number of
        clusters to check for (inclusive)
        If kmax is None, kmax should be set to the maximum number
            of clusters possible
    iterations is a positive integer containing the maximum number
        of iterations for the EM algorithm
    tol is a non-negative float containing the tolerance for the EM
        algorithm
    verbose is a boolean that determines if the EM algorithm should
        print information to the standard output

    Returns: best_k, best_result, l, b, or None, None, None, None
        on failure
        best_k is the best value for k based on its BIC
        best_result is a tuple containing pi, m, S
            pi is a numpy.ndarray of shape (k,) containing the
                cluster priors for the best number of clusters
            m is a numpy.ndarray of shape (k, d) containing the
                centroid means for the best number of clusters
            S is a numpy.ndarray of shape (k, d, d) containing the
                covariance matrices for the best number of clusters
        l is a numpy.ndarray of shape (kmax - kmin + 1) containing
            the log likelihood for each cluster size tested
        b is a numpy.ndarray of shape (kmax - kmin + 1) containing
            the BIC value for each cluster size tested
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n
    if not isinstance(kmax, int) or kmax <= 0:
        return None, None, None, None
    if kmin >= kmax:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    ks = range(kmin, kmax + 1)
    num_k = len(ks)

    log_likelihoods = np.zeros(num_k)
    b = np.zeros(num_k)
    results = []

    for idx, k in enumerate(ks):
        pi, m, S, g, log_l = expectation_maximization(
            X, k, iterations, tol, verbose)
        if pi is None or m is None or S is None:
            return None, None, None, None

        results.append((pi, m, S))

        p = (k - 1) + k * d + k * d * (d + 1) / 2

        log_likelihoods[idx] = log_l
        b[idx] = p * np.log(n) - 2 * log_l

    best_idx = np.argmin(b)
    best_k = kmin + best_idx
    best_result = results[best_idx]

    return best_k, best_result, log_likelihoods, b
