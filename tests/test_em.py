"""Sanity checks for gmmlab.em.fit -- the full EM loop.

Run with: .venv/bin/pytest tests/test_em.py -v

Unlike the other test files, there is no "plain-loop reference" here: fit()
is already built entirely from pieces (responsibilities, m_step,
log_likelihood) that each have their own loop-equivalence tests. What's new
and worth testing AT THE LOOP LEVEL is the thing those individual functions
can't show on their own: that L(theta) actually never decreases across many
real EM runs (lessons/02, Section 2.6), and that it matches an independent
implementation (scikit-learn) end to end.
"""

import numpy as np
import pytest
from sklearn.mixture import GaussianMixture

from gmmlab.datasets import make_two_gaussians
from gmmlab.em import (
    NonMonotonicLikelihoodError,
    _apply_clamping,
    best_of,
    fit,
    fit_best_of,
    fit_many,
    is_degenerate,
)
from gmmlab.likelihood import log_likelihood
from gmmlab.matching import match_components
from gmmlab.responsibilities import responsibilities


def test_likelihood_never_decreases_across_many_conditions():
    # sweep separation (lessons/01, Section 1.7) from heavy overlap to clean
    # separation: this is exactly the kind of EM run RQ1 will do, so the
    # monotonicity guarantee needs to hold here specifically, not just on an
    # easy textbook case.
    rng = np.random.default_rng(0)
    for trial in range(50):
        n_per_component, k_count, d_count = 40, 3, 2
        c = rng.uniform(0.5, 6.0)
        true_mu = rng.normal(scale=3, size=(k_count, d_count))
        x = np.vstack(
            [
                true_mu[k] + rng.normal(size=(n_per_component, d_count)) * c
                for k in range(k_count)
            ]
        )
        result = fit(x, k_count, rng=np.random.default_rng(trial))
        diffs = np.diff(result.log_likelihood_history)
        assert diffs.min() > -1e-8, f"trial {trial} (c={c:.2f}): likelihood decreased"


def test_matches_sklearn_on_well_separated_data():
    rng = np.random.default_rng(1)
    n_per_component, k_count = 150, 2
    true_mu = np.array([[-6.0, 0.0], [6.0, 0.0]])
    x = np.vstack(
        [true_mu[k] + rng.normal(size=(n_per_component, 2)) for k in range(k_count)]
    )

    ours = fit(x, k_count, rng=np.random.default_rng(1))
    ours_order = np.argsort(ours.mu[:, 0])

    sk = GaussianMixture(n_components=k_count, random_state=1, reg_covar=1e-6).fit(x)
    sk_order = np.argsort(sk.means_[:, 0])

    np.testing.assert_allclose(ours.mu[ours_order], sk.means_[sk_order], atol=1e-3)
    np.testing.assert_allclose(
        ours.log_likelihood_history[-1], sk.score(x) * len(x), rtol=1e-6
    )


def test_result_is_internally_consistent():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(60, 2))
    result = fit(x, 3, rng=rng)
    np.testing.assert_allclose(result.pi.sum(), 1.0)
    assert result.mu.shape == (3, 2)
    assert result.Sigma.shape == (3, 2, 2)
    assert result.n_iter == len(result.log_likelihood_history) - 1
    assert result.n_iter >= 1


def test_survives_more_components_than_it_can_support():
    # K=10 on 15 points: components competing for almost no data, a stress
    # test for the Nk_safe / reg_covar guards built into mstep.py -- here
    # exercised through many real EM iterations, not a single isolated call.
    rng = np.random.default_rng(3)
    x = rng.normal(size=(15, 2))
    result = fit(x, 10, n_iter=200, rng=np.random.default_rng(3))
    assert np.isfinite(result.log_likelihood_history).all()
    assert np.diff(result.log_likelihood_history).min() > -1e-8
    np.testing.assert_allclose(result.pi.sum(), 1.0)


def test_most_seeds_recover_the_true_structure_but_not_all():
    # On well-separated data, EM does NOT reliably converge to the true
    # structure from every random initialization -- this is real EM behaviour
    # (lessons/01_latent_variables_and_gmm.md, Lesson 1 Q5(c): initialization
    # sensitivity), discovered here directly: seed 4 converges (monotonically
    # -- the guarantee still holds) to means near (0, 0) instead of (+-6, 0)
    # because BOTH of its starting means happened to land in the same true
    # cluster. This is exactly what RQ2 will study formally; asserting "every
    # seed converges correctly" would be asserting something false about EM.
    rng = np.random.default_rng(6)
    true_mu = np.array([[-6.0, 0.0], [6.0, 0.0]])
    x = np.vstack([true_mu[k] + rng.normal(size=(100, 2)) for k in range(2)])

    recovered = 0
    for seed in range(30):
        result = fit(x, 2, rng=np.random.default_rng(seed))
        # never decreases, even on the seeds that fail to recover the truth
        assert np.diff(result.log_likelihood_history).min() > -1e-8
        found = sorted(result.mu[:, 0])
        if abs(found[0] - (-6)) < 1 and abs(found[1] - 6) < 1:
            recovered += 1
    assert recovered >= 15  # most should, not all -- see the comment above


def test_reproducible_with_same_seed():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(50, 2))
    r1 = fit(x, 3, rng=np.random.default_rng(42))
    r2 = fit(x, 3, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(r1.mu, r2.mu)


def test_rejects_k_larger_than_n():
    x = np.zeros((3, 2))
    with pytest.raises(ValueError):
        fit(x, 5, rng=np.random.default_rng(0))


def _make_overlapping_dataset(rng, n_per_component=150):
    """3 components, overlapping, unequal (non-spherical) covariance -- a
    harder test of the FULL covariance M-step than the well-separated,
    identity-covariance case in test_matches_sklearn_on_well_separated_data."""
    true_mu = np.array([[-2.0, 0.0], [2.0, 1.5], [0.0, -2.5]])
    true_covs = [
        np.array([[2.0, 0.8], [0.8, 0.5]]),
        np.array([[0.6, -0.3], [-0.3, 1.5]]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    ]
    chunks = [
        true_mu[k]
        + rng.normal(size=(n_per_component, 2)) @ np.linalg.cholesky(true_covs[k]).T
        for k in range(3)
    ]
    return np.vstack(chunks)


def test_single_restart_can_land_in_a_worse_local_optimum():
    # Documents, with a fixed reproducible seed, that a single EM run on
    # overlapping data is NOT guaranteed to find the best solution -- this
    # is lessons/01 Q5(c) (initialization sensitivity), a second instance of
    # it (the first is in research_log/IDEAS.md), found here on purpose
    # rather than papered over.
    x = _make_overlapping_dataset(np.random.default_rng(0))
    unlucky = fit(x, 3, rng=np.random.default_rng(2))
    best = fit_best_of(x, 3, n_restarts=10, rng=np.random.default_rng(99))
    assert best.log_likelihood_history[-1] > unlucky.log_likelihood_history[-1] + 50


def test_best_of_restarts_matches_sklearn_on_overlapping_data():
    x = _make_overlapping_dataset(np.random.default_rng(0))

    ours = fit_best_of(x, 3, n_restarts=10, rng=np.random.default_rng(99))
    sk = GaussianMixture(n_components=3, random_state=2, reg_covar=1e-6).fit(x)

    perm = match_components(ours.mu, sk.means_)
    np.testing.assert_allclose(ours.mu, sk.means_[perm], atol=0.2)
    np.testing.assert_allclose(
        ours.log_likelihood_history[-1], sk.score(x) * len(x), atol=1.0
    )


def test_fit_many_returns_every_restart_and_best_of_matches_its_max():
    x = _make_overlapping_dataset(np.random.default_rng(0))
    results = fit_many(x, 3, n_restarts=10, rng=np.random.default_rng(99))
    assert len(results) == 10

    lls = [r.log_likelihood_history[-1] for r in results]
    best = fit_best_of(x, 3, n_restarts=10, rng=np.random.default_rng(99))
    # none of these 10 restarts are degenerate, so the filter changes nothing
    # here -- fit_best_of must still agree with the plain max
    assert not any(is_degenerate(r) for r in results)
    assert best.log_likelihood_history[-1] == max(lls)


def test_unbounded_likelihood_pathology_is_detected_and_avoided():
    # found while designing RQ1 (research_log/IDEAS.md has the full numbers):
    # given enough iterations, EM on overlapping data can find a HIGHER
    # log-likelihood by collapsing one component onto a near-singular sliver
    # of a few points, instead of recovering the true 2-cluster structure.
    # lessons/02_jensen_and_elbo.md, Section 2.6 already named this (the likelihood
    # is unbounded above); this is that pathology actually occurring.
    x, z_true, mu_true = make_two_gaussians(
        c=2.0, sigma=1.0, n_per_component=300, rng=np.random.default_rng(0)
    )

    # the plain (undetected) global best, with enough iterations to find it,
    # IS the degenerate collapse -- confirms the pathology is real, not
    # hypothetical, on this exact dataset
    raw_best = max(
        fit_many(x, 2, n_restarts=10, n_iter=2000, rng=np.random.default_rng(1)),
        key=lambda r: r.log_likelihood_history[-1],
    )
    assert is_degenerate(raw_best)
    assert raw_best.pi.min() < 0.01

    # fit_best_of must sidestep it and return something close to the truth
    guarded = fit_best_of(
        x, 2, n_restarts=10, n_iter=2000, rng=np.random.default_rng(1)
    )
    assert not is_degenerate(guarded)
    recovered = sorted(guarded.mu[:, 0].tolist())
    assert abs(recovered[0] - (-1.0)) < 0.5
    assert abs(recovered[1] - 1.0) < 0.5


def test_a_real_decrease_raises_instead_of_being_called_converged(monkeypatch):
    # Engineers a FAKE log_likelihood that decreases on purpose, to prove the
    # detection logic itself works -- rather than relying on finding or
    # manufacturing a genuine E-step/M-step bug. Patches the name as em.py
    # imports it (`from .likelihood import log_likelihood`), which is what
    # fit() actually calls.
    import gmmlab.em as em_module

    sequence = iter([-1000.0, -950.0, -1050.0])  # 3rd call: a real 100-point drop
    monkeypatch.setattr(em_module, "log_likelihood", lambda *a, **k: next(sequence))

    x, _, _ = make_two_gaussians(
        c=4.0, n_per_component=20, rng=np.random.default_rng(0)
    )
    with pytest.raises(NonMonotonicLikelihoodError):
        fit(x, 2, n_iter=5, rng=np.random.default_rng(1))


def test_fit_many_skips_a_restart_that_raises_nonmonotonic(monkeypatch):
    # fit_many's whole point is robustness across restarts (is_degenerate
    # already filters out ones that converge to a collapsed answer); this
    # extends that to a restart that fails to converge at all. Engineers a
    # restart that raises, to prove fit_many skips exactly that one and
    # keeps the rest, rather than letting it crash the whole ensemble.
    import gmmlab.em as em_module

    x, _, _ = make_two_gaussians(
        c=4.0, n_per_component=20, rng=np.random.default_rng(0)
    )
    real_fit = em_module.fit
    call_count = [0]

    def flaky_fit(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 2:
            raise NonMonotonicLikelihoodError("engineered failure for this test")
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(em_module, "fit", flaky_fit)
    results = fit_many(x, 2, n_restarts=5, rng=np.random.default_rng(1))
    assert len(results) == 4  # exactly the one flaky restart was skipped


def test_fit_many_raises_if_every_restart_fails(monkeypatch):
    # if ALL restarts fail, that's no longer "one unlucky seed" -- fit_many
    # should say so loudly (RuntimeError) rather than silently return an
    # empty, useless list.
    import gmmlab.em as em_module

    x, _, _ = make_two_gaussians(
        c=4.0, n_per_component=20, rng=np.random.default_rng(0)
    )

    def always_fails(*args, **kwargs):
        raise NonMonotonicLikelihoodError("engineered failure for this test")

    monkeypatch.setattr(em_module, "fit", always_fails)
    with pytest.raises(RuntimeError):
        fit_many(x, 2, n_restarts=3, rng=np.random.default_rng(1))


def test_is_degenerate_threshold():
    x, _, _ = make_two_gaussians(
        c=6.0, n_per_component=100, rng=np.random.default_rng(0)
    )
    healthy = fit(x, 2, rng=np.random.default_rng(1))
    assert not is_degenerate(healthy)


def test_invalid_init_raises():
    x, _, _ = make_two_gaussians(
        c=4.0, n_per_component=20, rng=np.random.default_rng(0)
    )
    with pytest.raises(ValueError):
        fit(x, 2, rng=np.random.default_rng(1), init="bogus")


def test_default_init_is_unchanged_random():
    # the init= refactor must not silently change the long-standing default
    x, _, _ = make_two_gaussians(
        c=4.0, n_per_component=20, rng=np.random.default_rng(0)
    )
    default = fit(x, 2, rng=np.random.default_rng(1))
    explicit = fit(x, 2, rng=np.random.default_rng(1), init="random")
    np.testing.assert_array_equal(default.mu, explicit.mu)


def test_kmeanspp_rarely_starts_both_means_on_the_same_true_side():
    # found and verified (500 trials) while building RQ2
    # (research_log/RQ2_initialization.md): random init puts both starting
    # means on the same true side 49% of the time; kmeans++ should be rare.
    import gmmlab.em as em_module

    x, _, mu_true = make_two_gaussians(
        c=6.0, n_per_component=150, rng=np.random.default_rng(0)
    )

    same_side_count = 0
    trials = 100
    for seed in range(trials):
        _, mu, _, _ = em_module._init_kmeanspp(x, 2, 1e-6, np.random.default_rng(seed))
        same_side_count += (mu[0, 0] > 0) == (mu[1, 0] > 0)

    assert same_side_count < 25  # well below random init's ~49%


def test_best_of_matches_fit_best_of_without_recomputing():
    # best_of() on an already-computed fit_many() list must give the exact
    # same answer as fit_best_of() -- it's the same selection logic, just
    # applied to results you already have instead of recomputing them.
    x, _, _ = make_two_gaussians(
        c=1.0, n_per_component=100, rng=np.random.default_rng(0)
    )
    results = fit_many(x, 2, n_restarts=10, rng=np.random.default_rng(1))
    picked = best_of(results)
    direct = fit_best_of(x, 2, n_restarts=10, rng=np.random.default_rng(1))
    assert picked.log_likelihood_history[-1] == direct.log_likelihood_history[-1]


def test_apply_clamping_overwrites_only_clamped_rows():
    # lessons/04_clamped_em_and_active_learning.md, Section 4.1: a clamped point's
    # row becomes one-hot at its true label; everything else is untouched.
    r = np.array([[0.6, 0.4], [0.5, 0.5], [0.1, 0.9]])
    clamped_labels = np.array([-1, 1, -1])  # only the middle point is clamped
    out = _apply_clamping(r, clamped_labels)
    np.testing.assert_array_equal(out[0], [0.6, 0.4])  # untouched
    np.testing.assert_array_equal(out[1], [0.0, 1.0])  # clamped to component 1
    np.testing.assert_array_equal(out[2], [0.1, 0.9])  # untouched


def test_init_params_warm_starts_instead_of_using_init_strategy():
    # if init_params is given, fit() must start from THOSE exact parameters,
    # not from a fresh call to _init_random/_init_kmeanspp -- the whole point
    # of warm-starting (RQ3: continue from the existing model after a label
    # is revealed, instead of re-initializing from scratch).
    x, _, _ = make_two_gaussians(
        c=1.0, n_per_component=100, rng=np.random.default_rng(0)
    )
    # n_iter=2000 to genuinely reach tol, matching RQ1/RQ2's practice on this
    # kind of overlapping data (the default n_iter=100 isn't always enough).
    already_converged = fit(x, 2, n_iter=2000, rng=np.random.default_rng(1))
    assert already_converged.converged
    pi, mu, Sigma = already_converged.pi, already_converged.mu, already_converged.Sigma

    warm_started = fit(x, 2, n_iter=20, init_params=(pi, mu, Sigma))

    # starting already at a fixed point: the very first entry in the history
    # must match what these exact params produce, and it should converge
    # almost immediately, not drift off to something else.
    assert warm_started.log_likelihood_history[0] == pytest.approx(
        log_likelihood(x, pi, mu, Sigma)
    )
    assert warm_started.converged
    assert warm_started.n_iter <= 2


def test_clamped_point_stays_hard_assigned_after_fitting():
    # after fitting with one point clamped, its responsibility under the
    # FINAL parameters must still be one-hot at its true label -- clamping
    # is permanent for that point, not just a one-off nudge.
    x, z_true, _ = make_two_gaussians(
        c=1.0, n_per_component=100, rng=np.random.default_rng(0)
    )
    clamped_labels = np.full(len(x), -1)
    clamped_labels[0] = int(z_true[0])  # clamp just the first point, correctly

    result = fit(x, 2, rng=np.random.default_rng(1), clamped_labels=clamped_labels)
    r_final = responsibilities(x, result.pi, result.mu, result.Sigma)
    r_final_clamped = _apply_clamping(r_final, clamped_labels)
    np.testing.assert_array_equal(r_final_clamped[0], np.eye(2)[clamped_labels[0]])


def test_clamped_fit_does_not_raise_on_a_real_decrease(monkeypatch):
    # the standard monotonicity guarantee (lessons/02, Section 2.6) is proven for
    # the fully-unlabeled E-step; it does not automatically extend once some
    # responsibilities are fixed rather than recomputed (Lesson 4). A real
    # drop under clamping must NOT raise NonMonotonicLikelihoodError -- same
    # engineered-decrease trick as
    # test_a_real_decrease_raises_instead_of_being_called_converged, but this
    # time with clamped_labels set, where it must NOT raise.
    import gmmlab.em as em_module

    sequence = iter([-1000.0, -950.0, -1050.0, -1050.0])
    monkeypatch.setattr(em_module, "log_likelihood", lambda *a, **k: next(sequence))

    x, _, _ = make_two_gaussians(
        c=4.0, n_per_component=20, rng=np.random.default_rng(0)
    )
    clamped_labels = np.full(len(x), -1)
    clamped_labels[0] = 0
    result = fit(
        x, 2, n_iter=5, rng=np.random.default_rng(1), clamped_labels=clamped_labels
    )
    assert result is not None  # reaching here at all means no raise
