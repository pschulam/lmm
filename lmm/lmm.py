"""Implementation of linear mixed models."""

import logging

import numpy as np

from numpy.linalg import inv, solve
from scipy.stats import multivariate_normal as mvn


class LMM:
    def __init__(self, n_features, noise_var=None):
        self.mean = np.zeros(n_features)
        self.ranef_cov = np.eye(n_features)

        if noise_var is None:
            self._fit_noise = True
            self.noise = 1.0

        else:
            self._fit_noise = False
            self.noise = noise_var

    def log_likelihood(self, X, y):
        m, C, v = self.mean, self.ranef_cov, self.noise
        m_y = np.dot(X, m)
        C_y = np.dot(X, np.dot(C, X.T)) + v*np.eye(len(y))
        return mvn.logpdf(y, m_y, C_y)

    def coeff_posterior(self, X, y):
        m, C, v = self.mean, self.ranef_cov, self.noise
        w_prec = inv(C) + np.dot(X.T, X) / v
        w_mean = solve(w_prec, solve(C, m) + np.dot(X.T, y) / v)
        return w_mean, inv(w_prec)

    def fit(self, Xy_pairs, max_iter=500, tol=1e-5):
        self.mean[:] = block_lstsq(Xy_pairs)

        def objective():
            return sum(self.log_likelihood(*Xy) for Xy in Xy_pairs)

        logl = objective()
        converged = False
        num_iter = 0
        msg = '[lmm] iter={:04d}, avg_logl={:.2f}, delta={:.2e}'

        while not converged:
            old_logl = logl
            self.mean, self.ranef_cov, self.noise = self._em_step(Xy_pairs)
            logl = objective()

            num_iter += 1
            delta = (logl - old_logl) / np.abs(old_logl)

            m = msg.format(num_iter, logl, delta)
            logging.info(m)

            if delta < tol or num_iter >= max_iter:
                converged = True

    def _em_step(self, Xy_pairs):
        num_pairs = 0.0
        num_obs = 0.0

        m_agg = 0.0
        mm_agg = 0.0
        rss_agg = 0.0

        for X, y in Xy_pairs:
            num_pairs += 1
            num_obs += len(y)

            m, S = self.coeff_posterior(X, y)
            m_agg += m
            mm_agg += np.outer(m, m) + S

            r = y - np.dot(X, m)
            rss_agg += np.sum(r**2)
            rss_agg += np.sum(S * np.dot(X.T, X))

        mean = m_agg / num_pairs
        ranef_cov = mm_agg / num_pairs - np.outer(mean, mean)
        v = rss_agg / num_obs if self._fit_noise else self.noise

        return mean, ranef_cov, v


def block_lstsq(Xy_pairs):
    xtx = 0.0
    xty = 0.0

    for X, y in Xy_pairs:
        xtx += np.dot(X.T, X)
        xty += np.dot(X.T, y)

    return np.linalg.solve(xtx, xty)
