#!/usr/bin/env python3
"""
This module contains a function that performs agglomerative
clustering on a dataset using Ward linkage.
"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Performs agglomerative clustering on a dataset

    X is a numpy.ndarray of shape (n, d) containing the dataset
    dist is the maximum cophenetic distance for all clusters

    Performs agglomerative clustering with Ward linkage
    Displays the dendrogram with each cluster displayed in a
        different color

    Returns: clss, a numpy.ndarray of shape (n,) containing the
        cluster indices for each data point
    """
    linkage = scipy.cluster.hierarchy.linkage(X, method='ward')

    clss = scipy.cluster.hierarchy.fcluster(
        linkage, t=dist, criterion='distance')

    scipy.cluster.hierarchy.dendrogram(
        linkage, color_threshold=dist)
    plt.show()

    return clss
