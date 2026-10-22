"""Sanity checks for gmmlab.likelihood.log_likelihood.

Run with: .venv/bin/pytest tests/test_likelihood.py -v
"""

import math

import numpy as np
import pytest

from gmmlab.gaussian import gaussian_logpdf
from gmmlab.likelihood import log_likelihood


def log_likelihood_loop(X, pi, mu, Sigma):
    """Plain-loop reference, straight off the lessons/01, Section 1.6 formula."""
    n_count, k_count = X.shape[0], len(pi)
    total = 0.0
    for n in range(n_count):
        px = sum(
            pi[k] * math.exp(gaussian_logpdf(X[n : n + 1], mu[k], Sigma[k])[0])
            for k in range(k_count)
        )
        total += math.log(px)
    return total


def _random_mixture(rng, k_count=3, d_count=2):
    pi = rng.dirichlet(np.ones(k_count))
    mu = rng.normal(scale=2, size=(k_count, d_count))
    sigma = np.stack([np.eye(d_count) * 0.5 for _ in range(k_count)])
    return pi, mu, sigma


def test_matches_plain_loop():
    rng = np.random.default_rng(0)
    pi, mu, sigma = _random_mixture(rng)
    x = rng.normal(size=(12, 2))
    assert log_likelihood(x, pi, mu, sigma) == pytest.approx(
        log_likelihood_loop(x, pi, mu, sigma)
    )


def test_single_point_matches_gaussian_logpdf_directly():
    # K=1: the GMM log-likelihood of one point under one component is just
    # that component's own log-density -- no mixture machinery involved.
    mu = np.array([[1.0, -1.0]])
    sigma = np.array([np.eye(2) * 2.0])
    pi = np.array([1.0])
    x = np.array([[0.3, 0.1]])
    expected = gaussian_logpdf(x, mu[0], sigma[0])[0]
    assert log_likelihood(x, pi, mu, sigma) == pytest.approx(expected)


def test_additive_over_points():
    # L(theta) sums over independent points: splitting X into two chunks
    # and adding their separate log-likelihoods must equal the whole.
    rng = np.random.default_rng(1)
    pi, mu, sigma = _random_mixture(rng)
    x = rng.normal(size=(20, 2))
    whole = log_likelihood(x, pi, mu, sigma)
    parts = log_likelihood(x[:8], pi, mu, sigma) + log_likelihood(x[8:], pi, mu, sigma)
    assert whole == pytest.approx(parts)
