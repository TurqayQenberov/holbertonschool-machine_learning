#!/usr/bin/env python3
"""
Module that calculates the symmetric P affinities of a data set for
t-SNE.
"""
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Calculates the symmetric P affinities of a data set.

    X is a numpy.ndarray of shape (n, d) containing the dataset to be
        transformed by t-SNE
        n is the number of data points
        d is the number of dimensions in each point
    perplexity is the perplexity that all Gaussian distributions
        should have
    tol is the maximum tolerance allowed (inclusive) for the
        difference in Shannon entropy from perplexity for all
        Gaussian distributions

    Returns: P, a numpy.ndarray of shape (n, n) containing the
        symmetric P affinities
    """
    n, d = X.shape

    D, P, betas, H = P_init(X, perplexity)

    for i in range(n):
        low = None
        high = None

        Di = np.delete(D[i], i)

        Hi, Pi = HP(Di, betas[i])
        Hdiff = Hi - H

        while np.abs(Hdiff) > tol:
            if Hdiff > 0:
                low = betas[i, 0]
                if high is None:
                    betas[i, 0] *= 2
                else:
                    betas[i, 0] = (betas[i, 0] + high) / 2
            else:
                high = betas[i, 0]
                if low is None:
                    betas[i, 0] /= 2
                else:
                    betas[i, 0] = (betas[i, 0] + low) / 2

            Hi, Pi = HP(Di, betas[i])
            Hdiff = Hi - H

        P[i, np.arange(n) != i] = Pi

    P = (P + P.T) / (2 * n)

    return P
