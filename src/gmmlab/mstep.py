"""The M-step: closed-form parameter updates given fixed responsibilities.

Implements the three boxed formulas of lessons/03_m_step_and_lagrange.md, Section 3.5:

    pi_k        = N_k / N
    mu_k        = (1/N_k) * sum_n r_nk * x_n
    Sigma_k     = (1/N_k) * sum_n r_nk * (x_n - mu_k) (x_n - mu_k)^T

    where N_k = sum_n r_nk  (the "effective count" of component k; see Section 3.2)

Three separate functions, one per formula. mu_k has to be known before Sigma_k
can be computed, so sigma_step takes mu as an argument rather than
recomputing it. m_step() just calls the three in the right order.

Two things NOT in Lesson 3's math, added here because running code meets them
immediately (see the "Your turn" questions at the bottom of this file's
review):

  - `Nk_safe`: if a component ends up with zero responsibility everywhere
    (R[:, k] is all 0), N_k = 0 and mu_k, Sigma_k are literally 0/0. We clip
    the DENOMINATOR only, never N_k itself, so pi_k = N_k/N stays exactly 0,
    meaning the function returns 0 instead of nan. It does not "fix" a dead
    component (there is no sensible mu/Sigma for a component nothing was
    assigned to); it just keeps that failure from propagating as nan into
    every later computation. Whether and how often this happens is exactly
    what RQ2 (initialization sensitivity) measures.
  - `reg_covar`: added to every Sigma_k's diagonal unconditionally. Even an
    ALIVE component can have a near-singular Sigma_k, e.g. all its
    responsibility concentrated on very few points (see the review's
    near-singular test). This is the "covariance floor" promised in
    README's known gaps, and the name matches scikit-learn's
    GaussianMixture(reg_covar=...) on purpose, for the Stage 4 comparison.
"""

import numpy as np


def pi_step(R):
    """Mixing weights. lessons/03, Section 3.2. R: (N, K) -> pi: (K,)."""
    n_count = R.shape[0]
    return R.sum(axis=0) / n_count


def mu_step(X, R):
    """Component means. lessons/03, Section 3.3. X: (N, D), R: (N, K) -> mu: (K, D)."""
    n_k = R.sum(axis=0)
    n_k_safe = np.maximum(n_k, 1e-10)
    return (R.T @ X) / n_k_safe[:, None]


def sigma_step(X, R, mu, reg_covar=1e-6):
    """Component covariances. lessons/03, Section 3.4.

    X: (N, D), R: (N, K), mu: (K, D) -> Sigma: (K, D, D).
    """
    d_count = X.shape[1]
    n_k = R.sum(axis=0)
    n_k_safe = np.maximum(n_k, 1e-10)

    diff = X[:, None, :] - mu[None, :, :]  # (N, K, D)
    outer = diff[:, :, :, None] * diff[:, :, None, :]  # (N, K, D, D)
    sigma = (R[:, :, None, None] * outer).sum(axis=0) / n_k_safe[:, None, None]

    return sigma + reg_covar * np.eye(d_count)[None, :, :]


def m_step(X, R, reg_covar=1e-6):
    """Run all three updates. X: (N, D), R: (N, K) -> (pi, mu, Sigma)."""
    pi = pi_step(R)
    mu = mu_step(X, R)
    sigma = sigma_step(X, R, mu, reg_covar=reg_covar)
    return pi, mu, sigma
