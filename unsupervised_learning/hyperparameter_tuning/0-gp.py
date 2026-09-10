#!/usr/bin/env python3
"""Gaussian Process module."""
import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process."""
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Initializes the Gaussian Process.
        Args:
            X_init (numpy.ndarray): shape (t, 1), inputs already
                sampled with the black-box function.
            Y_init (numpy.ndarray): shape (t, 1), outputs of the
                black-box function for each input in X_init.
            l (float): the length parameter for the kernel.
            sigma_f (float): the standard deviation given to the
                output of the black-box function.
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """Calculates the covariance kernel matrix between two
        matrices using the Radial Basis Function (RBF).
        Args:
            X1 (numpy.ndarray): shape (m, 1).
            X2 (numpy.ndarray): shape (n, 1).
        Returns:
            numpy.ndarray: shape (m, n), the covariance kernel matrix.
        """
        sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) \
            + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * sqdist)
