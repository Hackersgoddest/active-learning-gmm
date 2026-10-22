"""Sanity checks for gmmlab.mstep.

Run with: .venv/bin/pytest tests/test_mstep.py -v
"""

import numpy as np
import pytest

from gmmlab.mstep import m_step, mu_step, pi_step, sigma_step


def m_step_loop(X, R, reg_covar=1e-6):
    """Plain-loop reference, straight off the three boxed formulas in Lesson 3."""
    n_count, d_count = X.shape
    k_count = R.shape[1]

    n_k = [sum(R[n, k] for n in range(n_count)) for k in range(k_count)]
    pi = [n_k[k] / n_count for k in range(k_count)]

    mu = []
    for k in range(k_count):
        s = np.zeros(d_count)
        for n in range(n_count):
            s += R[n, k] * X[n]
        mu.append(s / n_k[k] if n_k[k] > 0 else np.zeros(d_count))

    sigma = []
    for k in range(k_count):
        s = np.zeros((d_count, d_count))
        for n in range(n_count):
            d = X[n] - mu[k]
            s += R[n, k] * np.outer(d, d)
        s = s / n_k[k] if n_k[k] > 0 else np.zeros((d_count, d_count))
        sigma.append(s + reg_covar * np.eye(d_count))

    return np.array(pi), np.array(mu), np.array(sigma)


def _random_responsibilities(rng, n_count, k_count):
    logits = rng.normal(size=(n_count, k_count))
    r = np.exp(logits - logits.max(axis=1, keepdims=True))
    return r / r.sum(axis=1, keepdims=True)


def test_matches_plain_loop():
    rng = np.random.default_rng(0)
    n_count, k_count, d_count = 40, 3, 2
    x = rng.normal(size=(n_count, d_count))
    r = _random_responsibilities(rng, n_count, k_count)

    pi, mu, sigma = m_step(x, r)
    pi_l, mu_l, sigma_l = m_step_loop(x, r)
    np.testing.assert_allclose(pi, pi_l)
    np.testing.assert_allclose(mu, mu_l)
    np.testing.assert_allclose(sigma, sigma_l)


def test_pi_sums_to_one_and_matches_effective_counts():
    rng = np.random.default_rng(1)
    r = _random_responsibilities(rng, 25, 4)
    pi = pi_step(r)
    assert pi.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(pi * 25, r.sum(axis=0))  # pi_k = N_k / N


def test_sigma_is_symmetric_positive_definite():
    rng = np.random.default_rng(2)
    n_count, k_count, d_count = 30, 3, 3
    x = rng.normal(size=(n_count, d_count))
    r = _random_responsibilities(rng, n_count, k_count)
    mu = mu_step(x, r)
    sigma = sigma_step(x, r, mu)
    for k in range(k_count):
        np.testing.assert_allclose(sigma[k], sigma[k].T)
        assert np.linalg.eigvalsh(sigma[k]).min() > 0


def test_hard_assignment_reduces_to_ordinary_mean_and_covariance():
    # R one-hot (hard labels) should give the plain per-group sample mean/cov --
    # this is the Lesson 1, Section 1.6 complete-data MLE, recovered as a special case.
    rng = np.random.default_rng(3)
    group_a = rng.normal(loc=[-5, 0], size=(20, 2))
    group_b = rng.normal(loc=[5, 0], size=(20, 2))
    x = np.vstack([group_a, group_b])
    r = np.zeros((40, 2))
    r[:20, 0] = 1.0
    r[20:, 1] = 1.0

    pi, mu, sigma = m_step(x, r)
    np.testing.assert_allclose(pi, [0.5, 0.5])
    np.testing.assert_allclose(mu[0], group_a.mean(axis=0), atol=1e-10)
    np.testing.assert_allclose(mu[1], group_b.mean(axis=0), atol=1e-10)
    np.testing.assert_allclose(
        sigma[0], np.cov(group_a.T, bias=True) + 1e-6 * np.eye(2), atol=1e-10
    )


def test_dead_component_returns_zero_not_nan():
    # a component with zero responsibility everywhere: pi_k = 0 exactly
    # (correct), mu_k/Sigma_k become a defined placeholder instead of nan.
    rng = np.random.default_rng(4)
    x = rng.normal(size=(20, 2))
    r = _random_responsibilities(rng, 20, 3)
    r[:, 2] = 0.0
    r = r / r.sum(axis=1, keepdims=True)

    pi, mu, sigma = m_step(x, r)
    assert pi[2] == 0.0
    assert np.isfinite(mu[2]).all()
    assert np.isfinite(sigma[2]).all()
    np.testing.assert_allclose(np.linalg.eigvalsh(sigma[2]), [1e-6, 1e-6])


def test_reg_covar_keeps_near_singular_component_positive_definite():
    # ALIVE component, but almost all its responsibility sits on one point:
    # without the floor this Sigma_k would be near-singular.
    rng = np.random.default_rng(5)
    x = rng.normal(size=(30, 2))
    r = np.full((30, 2), 1e-8)
    r[0, 0] = 1.0
    r[:, 1] = 1.0
    r = r / r.sum(axis=1, keepdims=True)

    _, mu, sigma = m_step(x, r, reg_covar=1e-6)
    eigvals = np.linalg.eigvalsh(sigma[0])
    # the un-floored covariance is positive SEMI-definite (it's a weighted sum
    # of outer products), so its eigenvalues are >= 0; adding reg_covar*I
    # shifts every eigenvalue up by exactly reg_covar, so the floor alone
    # guarantees this lower bound regardless of how small the genuine spread is
    assert eigvals.min() >= 1e-6
    # and it shouldn't be much bigger: this component's real spread is tiny,
    # so the floor should dominate, not get swamped by it
    assert eigvals.min() < 100 * 1e-6
