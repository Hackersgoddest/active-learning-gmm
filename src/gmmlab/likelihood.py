"""The log-likelihood of the data under the current parameters.

Implements lessons/01_latent_variables_and_gmm.md, Section 1.6:

    L(theta) = sum_n [ ln( sum_k [ pi_k * N(x_n | mu_k, Sigma_k) ] ) ]

Separate from responsibilities.py on purpose, even though both build the same
log(pi_k) + log N_k(x) terms: responsibilities NORMALIZES those terms (divides
by their own logsumexp, per point); this SUMS the logsumexp itself, across
points. These are different reductions of the same intermediate quantity, kept
as two small, independently testable functions rather than one function
returning both, per the project's "small pieces, one thing each" habit.
"""

import numpy as np

from .gaussian import gaussian_logpdf
from .logsumexp import logsumexp


def log_likelihood(X, pi, mu, Sigma):
    """L(theta): a single number, how well (pi, mu, Sigma) explains ALL of X.

    X: (N, D), pi: (K,), mu: (K, D), Sigma: (K, D, D) -> a scalar.

    This is exactly what lessons/02_jensen_and_elbo.md, Section 2.6 proves EM never
    decreases: fit() below records this value every iteration specifically so
    that guarantee can be checked in running code, not just asserted in prose.
    """
    k_count = len(pi)
    log_weighted = np.stack(
        [np.log(pi[k]) + gaussian_logpdf(X, mu[k], Sigma[k]) for k in range(k_count)],
        axis=1,
    )  # (N, K)
    return logsumexp(log_weighted, axis=1).sum()  # (N,) -> scalar
