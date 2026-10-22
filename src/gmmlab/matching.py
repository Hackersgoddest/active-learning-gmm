"""Matching components between two GMM fits of the same data.

Closes the "label switching" item in README's known gaps: component k in one
fit and component k in another (a different seed, a different library, the
ground truth) have no reason to be the SAME physical cluster, since a GMM's
likelihood is unchanged by any relabeling of its K components (there are K!
equivalent relabelings of the same model). Comparing fit A's mu[0] to fit B's
mu[0] directly is only meaningful once they've been matched to the same
physical clusters first.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_components(mu_a, mu_b):
    """Find which component of `mu_b` corresponds to each component of `mu_a`.

    mu_a, mu_b: (K, D), matched by overall nearest mean. This minimizes the
    TOTAL distance summed over all K pairs (the Hungarian algorithm, via
    scipy.optimize.linear_sum_assignment), not each component's own nearest
    neighbor taken greedily one at a time. Those two are NOT the same thing:
    the greedy version can lock in a near-tie for one component and strand
    another component with a far worse partner than necessary, because the
    choice that's best FOR ONE PAIR isn't always best for the SUM. See
    tests/test_matching.py for a constructed case where greedy's total cost
    is ~34% worse than the optimal one found here.

    Returns
    -------
    perm : ndarray, shape (K,)
        `mu_b[perm]` is `mu_b`, reordered to line up with `mu_a` component by
        component.
    """
    cost = np.linalg.norm(mu_a[:, None, :] - mu_b[None, :, :], axis=2)  # (K, K)
    _, perm = linear_sum_assignment(cost)
    return perm
