"""Sanity checks for gmmlab.datasets.make_two_gaussians.

Run with: .venv/bin/pytest tests/test_datasets.py -v
"""

import numpy as np
import pytest

from gmmlab.datasets import make_two_gaussians


@pytest.mark.parametrize("c", [0.0, 0.5, 2.0, 6.0])
def test_true_separation_is_exactly_c(c):
    # the RETURNED mu is exact by construction -- not an empirical estimate
    _, _, mu = make_two_gaussians(c, sigma=1.5, n_per_component=5)
    assert np.linalg.norm(mu[0] - mu[1]) / 1.5 == pytest.approx(c)


def test_empirical_moments_converge_to_the_truth():
    x, z, mu = make_two_gaussians(
        c=4, sigma=1.0, n_per_component=50_000, rng=np.random.default_rng(1)
    )
    np.testing.assert_allclose(x[z == 0].mean(axis=0), mu[0], atol=0.02)
    np.testing.assert_allclose(x[z == 1].mean(axis=0), mu[1], atol=0.02)
    # per-component std, NOT the combined array's std -- the combined array
    # mixes two different means, so its overall spread is inflated by the
    # between-component distance, not just sigma (law of total variance)
    np.testing.assert_allclose(x[z == 0].std(axis=0), [1.0, 1.0], atol=0.02)
    np.testing.assert_allclose(x[z == 1].std(axis=0), [1.0, 1.0], atol=0.02)


def test_shapes():
    x, z, mu = make_two_gaussians(c=2, n_per_component=30)
    assert x.shape == (60, 2)
    assert z.shape == (60,)
    assert mu.shape == (2, 2)
    assert (z == 0).sum() == 30
    assert (z == 1).sum() == 30


def test_reproducible_with_same_seed():
    x1, z1, mu1 = make_two_gaussians(
        c=2, n_per_component=5, rng=np.random.default_rng(42)
    )
    x2, z2, mu2 = make_two_gaussians(
        c=2, n_per_component=5, rng=np.random.default_rng(42)
    )
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(z1, z2)
    np.testing.assert_array_equal(mu1, mu2)


def test_different_seeds_give_different_data():
    x1, *_ = make_two_gaussians(c=2, n_per_component=5, rng=np.random.default_rng(1))
    x2, *_ = make_two_gaussians(c=2, n_per_component=5, rng=np.random.default_rng(2))
    assert not np.array_equal(x1, x2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma": 0},
        {"sigma": -1},
        {"n_per_component": 0},
        {"n_per_component": -5},
        {"c": -3},  # found during ownership-question review: silently accepted before
    ],
)
def test_raises_on_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        make_two_gaussians(c=kwargs.pop("c", 2), **kwargs)


def test_unequal_split_gives_the_requested_imbalance():
    # research_log/RQ5b_true_imbalance.md needs the DATA itself imbalanced,
    # not just an equal split -- (n0, n1) in place of a single int.
    x, z, mu = make_two_gaussians(
        c=1.0, n_per_component=(540, 60), rng=np.random.default_rng(0)
    )
    assert (z == 0).sum() == 540
    assert (z == 1).sum() == 60
    assert x.shape == (600, 2)
    np.testing.assert_array_equal(
        mu, [[0.5, 0.0], [-0.5, 0.0]]
    )  # unaffected by the split


def test_unequal_split_rejects_a_zero_or_negative_count():
    with pytest.raises(ValueError):
        make_two_gaussians(c=1.0, n_per_component=(100, 0))
    with pytest.raises(ValueError):
        make_two_gaussians(c=1.0, n_per_component=(100, -5))


def test_combined_array_std_is_not_sigma_once_separated():
    # the trap: X.std(axis=0) on the COMBINED array is inflated by the
    # distance between the two true means (law of total variance), not just
    # sigma -- per-group std is what matches sigma. Locks in the mistake
    # found while verifying this function so it can't silently come back.
    x, z, _ = make_two_gaussians(
        c=4, sigma=1.0, n_per_component=50_000, rng=np.random.default_rng(1)
    )
    combined_std = x.std(axis=0)
    per_group_std = x[z == 0].std(axis=0)
    assert combined_std[0] > 2 * per_group_std[0]  # x-axis: badly inflated
    np.testing.assert_allclose(per_group_std, [1.0, 1.0], atol=0.02)
