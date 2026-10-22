"""Sanity checks for gmmlab.logsumexp.logsumexp.

Run with: .venv/bin/pytest tests/test_logsumexp.py -v
"""

import math

import numpy as np
import pytest
from scipy.special import logsumexp as scipy_logsumexp

from gmmlab.logsumexp import logsumexp


def logsumexp_loop(a):
    """Plain-loop reference: the readable specification of what we vectorize."""
    m = max(a)
    total = 0.0
    for x in a:
        total += math.exp(x - m)
    return m + math.log(total)


def test_matches_plain_loop():
    rng = np.random.default_rng(1)
    A = rng.normal(scale=20.0, size=(50, 6))
    expected = np.array([logsumexp_loop(list(row)) for row in A])
    np.testing.assert_allclose(logsumexp(A, axis=1), expected, atol=1e-12)


def test_hand_calculation():
    # ln(e^0 + e^0) = ln 2
    assert logsumexp(np.array([0.0, 0.0])) == pytest.approx(np.log(2))


def test_matches_scipy_and_axis_handling():
    rng = np.random.default_rng(0)
    A = rng.normal(scale=5.0, size=(7, 4, 3))
    for axis in (0, 1, 2, -1):
        np.testing.assert_allclose(
            logsumexp(A, axis=axis), scipy_logsumexp(A, axis=axis), atol=1e-12
        )
    np.testing.assert_allclose(
        logsumexp(A, axis=1, keepdims=True),
        scipy_logsumexp(A, axis=1, keepdims=True),
        atol=1e-12,
    )


def test_large_values_do_not_overflow():
    # naive: exp(1000) = inf
    with np.errstate(over="ignore"):
        assert np.exp(1000.0) == np.inf
    assert logsumexp(np.array([1000.0, 1000.0])) == pytest.approx(1000.0 + np.log(2))


def test_very_negative_values_do_not_underflow():
    # naive: exp(-1000) = 0.0, so ln(0+0) = -inf
    assert np.exp(-1000.0) == 0.0
    assert logsumexp(np.array([-1000.0, -1000.0])) == pytest.approx(-1000.0 + np.log(2))


def test_all_minus_inf_gives_minus_inf_not_nan():
    out = logsumexp(np.array([[-np.inf, -np.inf], [0.0, 0.0]]), axis=1)
    assert out[0] == -np.inf
    assert out[1] == pytest.approx(np.log(2))


def test_one_minus_inf_entry_is_ignored():
    # a zero-probability component contributes nothing
    assert logsumexp(np.array([-np.inf, 0.0])) == pytest.approx(0.0)
