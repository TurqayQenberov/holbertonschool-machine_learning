#!/usr/bin/env python3
"""
This module contains a function that calculates a Gaussian
Mixture Model from a dataset using sklearn.
"""
import sklearn.mixture


def gmm(X, k):
    """
    Calculates a GMM from a dataset

    X is a numpy.ndarray of shape (n, d) containing the dataset
    k is the number of clusters

    Returns: pi, m, S, clss, bic
        pi is a numpy.ndarray of shape (k,) containing the cluster
            priors
        m is a numpy.ndarray of shape (k, d) containing the
            centroid means
        S is a numpy.ndarray of shape (k, d, d) containing the
            covariance matrices
        clss is a numpy.ndarray of shape (n,) containing the
            cluster indices for each data point
        bic is the BIC value for the model
    """
    model = sklearn.mixture.GaussianMixture(n_components=k).fit(X)

    pi = model.weights_
    m = model.means_
    S = model.covariances_
    clss = model.predict(X)
    bic = model.bic(X)

    return pi, m, S, clss, bic
