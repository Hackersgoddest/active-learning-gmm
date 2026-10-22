"""Sanity checks for gmmlab.responsibilities.responsibilities.

Run with: .venv/bin/pytest tests/test_responsibilities.py -v
"""

import math

import numpy as np

from gmmlab.gaussian import gaussian_logpdf
from gmmlab.responsibilities import responsibilities


def responsibilities_loop(X, pi, mu, Sigma):
    """Plain-loop reference, straight off the Lesson 1 formula, no log-space trick."""
    n_count, k_count = X.shape[0], len(pi)
    r = np.zeros((n_count, k_count))
    for n in range(n_count):
        weights = [
            pi[k] * math.exp(gaussian_logpdf(X[n : n + 1], mu[k], Sigma[k])[0])
            for k in range(k_count)
        ]
        total = sum(weights)
        for k in range(k_count):
            r[n, k] = weights[k] / total
    return r


def _random_mixture(rng, k_count=3, d=2):
    pi = rng.dirichlet(np.ones(k_count))
    mu = rng.normal(scale=2, size=(k_count, d))
    sigma = np.stack([np.eye(d) * 0.5 for _ in range(k_count)])
    return pi, mu, sigma


def test_matches_plain_loop():
    rng = np.random.default_rng(0)
    pi, mu, sigma = _random_mixture(rng)
    x = rng.normal(size=(10, 2))
    np.testing.assert_allclose(
        responsibilities(x, pi, mu, sigma), responsibilities_loop(x, pi, mu, sigma)
    )


def test_rows_sum_to_one():
    rng = np.random.default_rng(1)
    pi, mu, sigma = _random_mixture(rng, k_count=4, d=3)
    x = rng.normal(size=(25, 3))
    r = responsibilities(x, pi, mu, sigma)
    np.testing.assert_allclose(r.sum(axis=1), np.ones(25), atol=1e-10)


def test_nonnegative():
    rng = np.random.default_rng(2)
    pi, mu, sigma = _random_mixture(rng)
    x = rng.normal(size=(30, 2))
    assert (responsibilities(x, pi, mu, sigma) >= 0).all()


def test_point_at_a_mean_favours_that_component():
    # a point placed exactly at mu_1, with components well separated, should
    # be assigned almost entirely to component 1
    pi = np.array([1 / 3, 1 / 3, 1 / 3])
    mu = np.array([[-10.0, 0.0], [0.0, 0.0], [10.0, 0.0]])
    sigma = np.stack([np.eye(2) for _ in range(3)])
    r = responsibilities(np.array([[0.0, 0.0]]), pi, mu, sigma)
    np.testing.assert_allclose(r, [[0.0, 1.0, 0.0]], atol=1e-6)


def test_far_point_does_not_become_nan():
    # every component's raw density underflows to exactly 0.0 here, so the
    # NAIVE (non-log-space) ratio is 0/0 = nan. Ours still returns a valid
    # distribution, assigning everything to the least-implausible component.
    pi = np.array([0.2, 0.3, 0.5])
    mu = np.array([[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    sigma = np.stack([np.eye(2) * 0.5 for _ in range(3)])
    far = np.array([[500.0, 500.0]])

    with np.errstate(all="ignore"):
        raw = np.array(
            [pi[k] * np.exp(gaussian_logpdf(far, mu[k], sigma[k])[0]) for k in range(3)]
        )
        assert (raw == 0.0).all()  # confirms the naive approach WOULD be 0/0

    r = responsibilities(far, pi, mu, sigma)
    assert np.isfinite(r).all()
    np.testing.assert_allclose(r.sum(), 1.0)


def test_equal_distance_gives_equal_responsibility():
    # a point equidistant from two identical, equally-weighted components:
    # by symmetry, responsibilities must split 50/50
    pi = np.array([0.5, 0.5])
    mu = np.array([[-1.0, 0.0], [1.0, 0.0]])
    sigma = np.stack([np.eye(2) for _ in range(2)])
    r = responsibilities(np.array([[0.0, 0.0]]), pi, mu, sigma)
    np.testing.assert_allclose(r, [[0.5, 0.5]])


def test_shape():
    rng = np.random.default_rng(3)
    pi, mu, sigma = _random_mixture(rng, k_count=5, d=2)
    x = rng.normal(size=(7, 2))
    assert responsibilities(x, pi, mu, sigma).shape == (7, 5)
