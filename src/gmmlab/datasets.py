"""Synthetic datasets for the research questions.

make_two_gaussians is the RQ1 data generator: the setup from
lessons/RQ1_prep_separation_and_bayes_error.md, turned into code. K=2, equal
weight, shared spherical covariance sigma^2 * I, controlled entirely by the
separation c (lessons/01, Section 1.7):

    c = ||mu_1 - mu_2|| / sigma

The two means are placed symmetrically on the x-axis, mu_1 = (c*sigma/2, 0)
and mu_2 = (-c*sigma/2, 0), matching the worked examples in the prep note
exactly, so numbers computed by hand there (e.g. responsibility at the
midpoint, at one step away) apply directly to data generated here.
"""

import numpy as np


def make_two_gaussians(c, sigma=1.0, n_per_component=100, rng=None):
    """Sample from two spherical Gaussians with separation c.

    Parameters
    ----------
    c : float
        Separation: ||mu_1 - mu_2|| / sigma. c=0 places both means at the
        same point (total overlap); larger c means further apart.
    sigma : float, default 1.0
        Shared standard deviation; covariance is sigma^2 * I for both.
    n_per_component : int or (int, int), default 100
        Points drawn from each component. A single int gives an EQUAL split
        (what every RQ before RQ5's imbalance angle used); a pair (n0, n1)
        gives an UNEQUAL split, added for research_log/RQ5b_true_imbalance.md,
        which needs the data itself imbalanced rather than just the model's
        belief about it (that was RQ5_blind_spot.md's angle).
    rng : numpy.random.Generator, optional
        For reproducibility: the same rng state gives the same data.

    Returns
    -------
    X : ndarray, shape (n0+n1, 2)
        All points, component 0's points first, then component 1's.
    z : ndarray, shape (n0+n1,)
        True labels: 0 for component 0's points, 1 for component 1's.
    mu : ndarray, shape (2, 2)
        The TRUE means used to generate the data (ground truth, for checking
        what EM recovers against).
    """
    if c < 0:
        raise ValueError(
            f"c is a separation (a distance / sigma) -- must be >= 0, got {c}"
        )
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    if np.isscalar(n_per_component):
        n0, n1 = int(n_per_component), int(n_per_component)
    else:
        n0, n1 = (int(n) for n in n_per_component)
    if n0 < 1 or n1 < 1:
        raise ValueError(
            f"n_per_component must be >= 1 for each component, got {n_per_component}"
        )

    rng = rng if rng is not None else np.random.default_rng()
    mu = np.array([[c * sigma / 2, 0.0], [-c * sigma / 2, 0.0]])

    x = np.vstack(
        [
            mu[0] + sigma * rng.normal(size=(n0, 2)),
            mu[1] + sigma * rng.normal(size=(n1, 2)),
        ]
    )
    z = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
    return x, z, mu
