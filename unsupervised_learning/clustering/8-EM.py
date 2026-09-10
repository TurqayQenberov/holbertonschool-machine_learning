#!/usr/bin/env python3
"""
This module contains a function that performs the full
Expectation-Maximization algorithm for a Gaussian Mixture Model.
"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5,
                             verbose=False):
    """
    Performs the expectation maximization for a GMM

    X is a numpy.ndarray of shape (n, d) containing the data set
    k is a positive integer containing the number of clusters
    iterations is a positive integer containing the maximum number
        of iterations for the algorithm
    tol is a non-negative float containing tolerance of the log
        likelihood, used to determine early stopping
    verbose is a boolean that determines if you should print
        information about the algorithm
        If True, print Log Likelihood after {i} iterations: {l}
            every 10 iterations and after the last iteration

    Returns: pi, m, S, g, l, or None, None, None, None, None on
        failure
        pi is a numpy.ndarray of shape (k,) containing the priors
            for each cluster
        m is a numpy.ndarray of shape (k, d) containing the
            centroid means for each cluster
        S is a numpy.ndarray of shape (k, d, d) containing the
            covariance matrices for each cluster
        g is a numpy.ndarray of shape (k, n) containing the
            probabilities for each data point in each cluster
        l is the log likelihood of the model
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    g, l_prev = expectation(X, pi, m, S)
    if g is None or l_prev is None:
        return None, None, None, None, None

    if verbose:
        print('Log Likelihood after 0 iterations: {}'.format(
            round(l_prev, 5)))

    for i in range(1, iterations + 1):
        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        g, l_curr = expectation(X, pi, m, S)
        if g is None or l_curr is None:
            return None, None, None, None, None

        diff = abs(l_curr - l_prev)
        l_prev = l_curr

        if verbose and (i % 10 == 0 or diff <= tol or i == iterations):
            print('Log Likelihood after {} iterations: {}'.format(
                i, round(l_curr, 5)))

        if diff <= tol:
            break

    return pi, m, S, g, l_prev
