#!/usr/bin/env python3
"""Bayesian Optimization module."""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimzation on a noiseless 1D Gaussian
    process.
    """
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """Initializes Bayesian Optimization.
        Args:
            f: the black-box function to be optimized.
            X_init (numpy.ndarray): shape (t, 1), inputs already
                sampled with the black-box function.
            Y_init (numpy.ndarray): shape (t, 1), outputs of the
                black-box function for each input in X_init.
            bounds (tuple): (min, max), the bounds of the space in
                which to look for the optimal point.
            ac_samples (int): the number of samples that should be
                analyzed during acquisition.
            l (float): the length parameter for the kernel.
            sigma_f (float): the standard deviation given to the
                output of the black-box function.
            xsi (float): the exploration-exploitation factor for
                acquisition.
            minimize (bool): determines whether optimization should
                be performed for minimization (True) or maximization
                (False).
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        min_b, max_b = bounds
        self.X_s = np.linspace(min_b, max_b, ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """Calculates the next best sample location using the
        Expected Improvement acquisition function.
        Returns:
            X_next (numpy.ndarray): shape (1,), the next best sample
                point.
            EI (numpy.ndarray): shape (ac_samples,), the expected
                improvement of each potential sample.
        """
        mu, sigma = self.gp.predict(self.X_s)
        if self.minimize:
            mu_sample_opt = np.min(self.gp.Y)
            imp = mu_sample_opt - mu - self.xsi
        else:
            mu_sample_opt = np.max(self.gp.Y)
            imp = mu - mu_sample_opt - self.xsi
        with np.errstate(divide='warn'):
            Z = imp / sigma
            EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            EI[sigma == 0.0] = 0.0
        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI

    def optimize(self, iterations=100):
        """Optimizes the black-box function.
        Args:
            iterations (int): the maximum number of iterations to
                perform.
        Returns:
            X_opt (numpy.ndarray): shape (1,), the optimal point.
            Y_opt (numpy.ndarray): shape (1,), the optimal function
                value.
        """
        for _ in range(iterations):
            X_next, _ = self.acquisition()
            if X_next in self.gp.X:
                break
            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)
        if self.minimize:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)
        X_opt = self.gp.X[idx]
        Y_opt = self.gp.Y[idx]
        return X_opt, Y_opt
