"""Sanity checks for gmmlab.gaussian.gaussian_logpdf.

Run with: .venv/bin/pytest tests/test_gaussian.py -v
"""

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from gmmlab.gaussian import gaussian_logpdf


def gaussian_logpdf_loop(x, mu, Sigma):
    """Plain-loop reference for ONE point, straight from the formula:
    -0.5 * (D ln(2 pi) + ln|Sigma| + (x-mu)^T Sigma^-1 (x-mu))."""
    D = len(mu)
    inv = np.linalg.inv(Sigma)
    quad = 0.0
    for i in range(D):
        for j in range(D):
            quad += (x[i] - mu[i]) * inv[i, j] * (x[j] - mu[j])
    return -0.5 * (D * np.log(2 * np.pi) + np.log(np.linalg.det(Sigma)) + quad)


def test_matches_plain_loop():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(3, 3))
    Sigma = A @ A.T + 3 * np.eye(3)
    mu = rng.normal(size=3)
    X = rng.normal(size=(12, 3))
    expected = np.array([gaussian_logpdf_loop(x, mu, Sigma) for x in X])
    np.testing.assert_allclose(gaussian_logpdf(X, mu, Sigma), expected, atol=1e-10)


def test_hand_calculation():
    # N(0 | 0, I) in D=2: the exp() term is 1, so this reduces to
    # -D/2 * ln(2*pi), computable without any code at all.
    D = 2
    val = gaussian_logpdf(np.zeros((1, D)), np.zeros(D), np.eye(D))[0]
    assert val == pytest.approx(-D / 2 * np.log(2 * np.pi))


@pytest.mark.parametrize("D", [1, 2, 3, 5])
def test_matches_scipy(D):
    rng = np.random.default_rng(0)
    for _ in range(20):
        A = rng.normal(size=(D, D))
        Sigma = A @ A.T + D * np.eye(D)  # guaranteed symmetric positive definite
        mu = rng.normal(size=D)
        X = rng.normal(size=(15, D))

        mine = gaussian_logpdf(X, mu, Sigma)
        theirs = multivariate_normal(mean=mu, cov=Sigma).logpdf(X)
        np.testing.assert_allclose(mine, theirs, atol=1e-8)


@pytest.mark.parametrize(
    "bad_Sigma",
    [
        np.array([[1.0, 2.0], [2.0, 1.0]]),  # eigenvalues 3, -1: det < 0
        -np.eye(2),  # eigenvalues -1, -1: det = +1 (sign test alone misses this)
        np.array([[2.0, 1.0], [0.0, 2.0]]),  # not symmetric, det > 0
    ],
)
def test_raises_on_invalid_covariance(bad_Sigma):
    with pytest.raises(np.linalg.LinAlgError):
        gaussian_logpdf(np.zeros((1, 2)), np.zeros(2), bad_Sigma)


def test_raises_on_shape_mismatch():
    # X has 3 columns, mu has length 2
    with pytest.raises(ValueError):
        gaussian_logpdf(np.zeros((4, 3)), np.zeros(2), np.eye(2))
    # a column-vector mu would silently make D = 1
    with pytest.raises(ValueError):
        gaussian_logpdf(np.zeros((4, 2)), np.zeros((1, 2)), np.eye(2))
    # X must be 2-D
    with pytest.raises(ValueError):
        gaussian_logpdf(np.zeros(2), np.zeros(2), np.eye(2))


def test_finite_where_raw_density_underflows():
    # At distance ~56 from mu the raw density is exactly 0.0 in float64, but
    # the log-density is an ordinary number. This is why we work in log-space.
    far = np.array([[40.0, 40.0]])
    val = gaussian_logpdf(far, np.zeros(2), np.eye(2))[0]
    assert np.isfinite(val)
    assert val == pytest.approx(-1600 - np.log(2 * np.pi))
