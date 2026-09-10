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

    def predict(self, X_s):
        """Predicts the mean and standard deviation of points in a
        Gaussian process.
        Args:
            X_s (numpy.ndarray): shape (s, 1), points whose mean and
                standard deviation should be calculated.
        Returns:
            mu (numpy.ndarray): shape (s,), the mean for each point
                in X_s.
            sigma (numpy.ndarray): shape (s,), the variance for each
                point in X_s.
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.pinv(self.K)
        mu = K_s.T.dot(K_inv).dot(self.Y).reshape(-1)
        sigma = np.diag(K_ss - K_s.T.dot(K_inv).dot(K_s))
        return mu, sigma

    def update(self, X_new, Y_new):
        """Updates a Gaussian Process with a new sample point.
        Args:
            X_new (numpy.ndarray): shape (1,), the new sample point.
            Y_new (numpy.ndarray): shape (1,), the new sample
                function value.
        """
        self.X = np.vstack((self.X, X_new.reshape(-1, 1)))
        self.Y = np.vstack((self.Y, Y_new.reshape(-1, 1)))
        self.K = self.kernel(self.X, self.X)
