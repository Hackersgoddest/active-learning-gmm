"""Responsibilities: the E-step.

Implements lessons/01_latent_variables_and_gmm.md, Section 1.4:

    r_nk = pi_k * N(x_n | mu_k, Sigma_k) / sum_j [ pi_j * N(x_n | mu_j, Sigma_j) ]

and lessons/02_jensen_and_elbo.md, Section 2.4: this IS the E-step, since computing
r_nk from the current parameters is exactly the q that makes the ELBO touch
the true log-likelihood.

Computed in log-space throughout (see numerator/denominator both as
ln(pi_k) + ln N_k(x_n), combined with logsumexp), for the same reason
gaussian_logpdf works in log-space: a point far from EVERY component makes
every raw density underflow to 0.0, and 0/0 is nan. The log-space version
still gives a sensible answer, something like "closest to the least
implausible component," instead of a crash.
"""

import numpy as np

from .gaussian import gaussian_logpdf
from .logsumexp import logsumexp


def responsibilities(X, pi, mu, Sigma):
    """Responsibility of every component for every point.

    Parameters
    ----------
    X : ndarray, shape (N, D)
    pi : ndarray, shape (K,)
        Mixing weights, pi_k >= 0, sum(pi) == 1.
    mu : ndarray, shape (K, D)
    Sigma : ndarray, shape (K, D, D)
        Sigma[k] symmetric positive definite (gaussian_logpdf checks this).

    Returns
    -------
    R : ndarray, shape (N, K)
        R[n, k] = r_nk. Every row sums to 1 (up to floating-point error).
    """
    k_count = len(pi)
    # log_weighted[n, k] = ln(pi_k) + ln N(x_n | mu_k, Sigma_k)
    log_weighted = np.stack(
        [np.log(pi[k]) + gaussian_logpdf(X, mu[k], Sigma[k]) for k in range(k_count)],
        axis=1,
    )  # (N, K)

    log_norm = logsumexp(log_weighted, axis=1, keepdims=True)  # (N, 1): ln p(x_n)
    return np.exp(log_weighted - log_norm)  # (N, K)
