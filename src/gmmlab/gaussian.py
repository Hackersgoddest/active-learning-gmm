"""Multivariate Gaussian log-density.

Implements lessons/01_latent_variables_and_gmm.md Section 1.2:

    N(x | mu, Sigma) = (2*pi)^(-D/2) * det(Sigma)^(-1/2)
                       * exp( -0.5 * (x-mu)^T Sigma^-1 (x-mu) )

We compute the LOG of this rather than the density itself. Every place that
calls this (responsibilities, the log-likelihood) works in log-space anyway
(see lessons/01, Section 1.4), and the raw density underflows to exactly 0.0 for
points even moderately far from mu, long before we'd want it to.
"""

import numpy as np


def gaussian_logpdf(X, mu, Sigma):
    """Log-density of N(mu, Sigma) at every row of X.

    Parameters
    ----------
    X : ndarray, shape (N, D)
        N points, each a D-dimensional row.
    mu : ndarray, shape (D,)
    Sigma : ndarray, shape (D, D)
        Symmetric positive definite.

    Returns
    -------
    logpdf : ndarray, shape (N,)

    Notes
    -----
    Two deliberate choices, both about not computing things we don't need:

    - `np.linalg.slogdet` gives ln|Sigma| directly. We never form det(Sigma)
      itself, which can overflow/underflow to inf/0 for larger D even when
      its log is a perfectly ordinary number.
    - `np.linalg.solve(Sigma, diff.T)` gives Sigma^-1 @ diff for every point
      without ever forming the explicit inverse matrix Sigma^-1. We only ever
      need Sigma^-1 applied to a vector, never the matrix on its own, and
      solving a linear system is both cheaper and numerically better
      conditioned than inverting and then multiplying.

    Validation is explicit because these failures would otherwise be SILENT
    (wrong number, no error):

    - A `mu` of shape (1, D) makes `mu.shape[0]` equal 1, so the D in the
      normalising constant is wrong while broadcasting still "works".
    - The sign of det(Sigma) does NOT test positive definiteness: Sigma = -I in
      D=2 has det = +1. Only all eigenvalues > 0 does.
    - A non-symmetric Sigma is not a covariance matrix; the formula would
      quietly evaluate something that is not a Gaussian density.

    (See README "known gaps": a covariance floor, added with the M-step, is
    what keeps Sigma positive definite during EM.)
    """
    if mu.ndim != 1:
        raise ValueError(f"mu must have shape (D,), got {mu.shape}")
    if X.ndim != 2:
        raise ValueError(f"X must have shape (N, D), got {X.shape}")
    D = mu.shape[0]
    if X.shape[1] != D:
        raise ValueError(f"X has {X.shape[1]} columns but mu has length {D}")
    if Sigma.shape != (D, D):
        raise ValueError(f"Sigma must have shape ({D}, {D}), got {Sigma.shape}")
    if not np.allclose(Sigma, Sigma.T):
        raise np.linalg.LinAlgError("Sigma is not symmetric")
    if np.linalg.eigvalsh(Sigma).min() <= 0:
        raise np.linalg.LinAlgError("Sigma is not positive definite")

    diff = X - mu  # (N, D)
    _, logdet = np.linalg.slogdet(Sigma)  # sign is +1 now, checked above

    solved = np.linalg.solve(Sigma, diff.T).T  # (N, D): Sigma^-1 @ diff, row-wise
    quad = np.einsum("nd,nd->n", diff, solved)  # (N,): (x_n-mu)^T Sigma^-1 (x_n-mu)

    return -0.5 * (D * np.log(2 * np.pi) + logdet + quad)
