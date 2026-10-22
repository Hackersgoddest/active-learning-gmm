"""Sanity checks for gmmlab.matching.match_components.

Run with: .venv/bin/pytest tests/test_matching.py -v
"""

import numpy as np

from gmmlab.matching import match_components


def _greedy_match(mu_a, mu_b):
    """The naive alternative we're deliberately NOT using -- kept here only
    to demonstrate why, in test_beats_greedy_matching below."""
    order, used = [], set()
    for a in mu_a:
        dists = [
            np.linalg.norm(a - b) if j not in used else np.inf
            for j, b in enumerate(mu_b)
        ]
        j = int(np.argmin(dists))
        used.add(j)
        order.append(j)
    return np.array(order)


def _total_cost(mu_a, mu_b, perm):
    return sum(np.linalg.norm(mu_a[i] - mu_b[perm[i]]) for i in range(len(mu_a)))


def test_exact_match_is_found():
    mu_a = np.array([[0.0, 0.0], [5.0, 5.0], [-3.0, 2.0]])
    perm = np.array([2, 0, 1])  # some scramble of mu_a
    mu_b = mu_a[perm]
    found = match_components(mu_a, mu_b)
    np.testing.assert_array_equal(mu_b[found], mu_a)


def test_beats_greedy_matching():
    # found by random search specifically for a case where greedy's choice
    # for the FIRST point forecloses a much better overall assignment
    mu_a = np.array([[7.148, -9.328], [4.593, -6.487], [7.264, 0.829]])
    mu_b = np.array([[-4.006, -1.546], [-9.434, -7.514], [3.412, 2.944]])

    optimal = match_components(mu_a, mu_b)
    greedy = _greedy_match(mu_a, mu_b)

    optimal_cost = _total_cost(mu_a, mu_b, optimal)
    greedy_cost = _total_cost(mu_a, mu_b, greedy)

    assert optimal_cost < greedy_cost  # the two methods really do disagree here
    assert not np.array_equal(optimal, greedy)


def test_permutation_is_valid():
    rng = np.random.default_rng(0)
    mu_a = rng.normal(size=(5, 3))
    mu_b = rng.normal(size=(5, 3))
    perm = match_components(mu_a, mu_b)
    assert sorted(perm.tolist()) == list(range(5))  # every index used exactly once
