"""The EM loop: alternate the E-step and M-step until the log-likelihood stops
improving.

This is Lesson 2 in executable form: Section 2.4 (E-step = compute responsibilities),
Section 2.5 (M-step = the closed-form updates), and Section 2.6 (the guarantee that
L(theta) never decreases), checked here every iteration via
`log_likelihood_history` rather than just asserted in the lesson text.
"Checked" means checked: a real decrease raises NonMonotonicLikelihoodError
instead of being folded into "converged" (an earlier version of this loop
stopped on `improvement < tol` alone, which is also true for a large
NEGATIVE improvement, so a 50-point drop from a bug would have been silently
labeled converged=True, identical to genuine convergence. Caught by review,
fixed below, and locked in by
tests/test_em.py::test_a_real_decrease_raises_instead_of_being_called_converged).

Two initialization strategies (`init="random"`, the long-standing default,
and `init="kmeans++"`, added for RQ2) both start from K distinct data points
as the means, uniform pi, and the overall data covariance shared by every
component; they differ only in HOW the K starting points are chosen.
Verified before writing it in: on well-separated data, random init puts both
starting points on the same true side 49% of the time (500 trials), while
kmeans++ drops that to 10% (research_log/RQ2_initialization.md has the
numbers this motivated).
"""

from dataclasses import dataclass, field

import numpy as np

from .likelihood import log_likelihood
from .mstep import m_step
from .responsibilities import responsibilities


@dataclass
class GMMFit:
    """Everything fit() produces: the parameters, and how it got there."""

    pi: np.ndarray  # (K,)
    mu: np.ndarray  # (K, D)
    Sigma: np.ndarray  # (K, D, D)
    log_likelihood_history: list = field(default_factory=list)
    converged: bool = False
    # Row indices into X that were chosen as the starting means, in the order
    # each init strategy chose them (e.g. init_indices[1] is k-means++'s
    # SECOND pick), needed to test RQ2's init-mechanism follow-up
    # (research_log/RQ2_initialization.md, Section 11): which data point an
    # initialization strategy picked, rather than just where EM ended up.
    init_indices: np.ndarray | None = None

    @property
    def n_iter(self):
        """EM iterations actually run (history includes the pre-fit value)."""
        return len(self.log_likelihood_history) - 1


def _pi_sigma_start(X, k_count, reg_covar):
    """The part of initialization shared by every strategy: uniform pi, the
    overall data covariance (plus the floor) copied to every component. Only
    how `mu` is chosen differs between strategies below."""
    d_count = X.shape[1]
    pi = np.full(k_count, 1.0 / k_count)
    base_cov = np.cov(X, rowvar=False).reshape(d_count, d_count)
    Sigma = np.stack([base_cov + reg_covar * np.eye(d_count) for _ in range(k_count)])
    return pi, Sigma


def _init_random(X, k_count, reg_covar, rng):
    """K data points chosen uniformly at random as the starting means."""
    n_count = X.shape[0]
    idx = rng.choice(n_count, size=k_count, replace=False)
    mu = X[idx].copy()
    pi, Sigma = _pi_sigma_start(X, k_count, reg_covar)
    return pi, mu, Sigma, idx


def _init_kmeanspp(X, k_count, reg_covar, rng):
    """K data points chosen by the k-means++ rule: the first uniformly at
    random, each next one with probability proportional to its SQUARED
    distance to the nearest already-chosen point, so points far from what's
    already picked are far more likely to be picked next. This directly
    targets the Stage 3 failure mode (both starting means landing on the
    same true side by chance): verified 500 trials, see module docstring.
    """
    n_count = X.shape[0]
    chosen = [int(rng.integers(n_count))]
    for _ in range(k_count - 1):
        dists_sq = np.min(
            [np.sum((X - X[i]) ** 2, axis=1) for i in chosen], axis=0
        )  # (N,): each point's distance to its NEAREST already-chosen center
        probs = dists_sq / dists_sq.sum()
        chosen.append(int(rng.choice(n_count, p=probs)))
    mu = X[chosen].copy()
    pi, Sigma = _pi_sigma_start(X, k_count, reg_covar)
    return pi, mu, Sigma, np.array(chosen)


_INIT_STRATEGIES = {"random": _init_random, "kmeans++": _init_kmeanspp}


class NonMonotonicLikelihoodError(RuntimeError):
    """Raised when log_likelihood DECREASED between EM iterations.

    lessons/02_jensen_and_elbo.md, Section 2.6 proves this can never happen for a
    correct E-step + M-step. If it happens anyway, that is an arithmetic bug
    in the E-step or M-step, not a modeling issue; the lesson says so
    explicitly. -1e-8 is the allowance for ordinary floating-point noise, not
    a "small decrease is fine" rule; it's the same threshold already used
    throughout tests/ to check this guarantee (e.g.
    test_likelihood_never_decreases_across_many_conditions), not a new number
    invented for this check.
    """


def _apply_clamping(r, clamped_labels):
    """Overwrite responsibility rows for clamped points with a one-hot vector
    at their true label (lessons/04_clamped_em_and_active_learning.md, Section 4.1).
    `clamped_labels`: (N,) int array, -1 for "not clamped" (left untouched)."""
    r = r.copy()
    mask = clamped_labels >= 0
    r[mask] = 0.0
    r[mask, clamped_labels[mask]] = 1.0
    return r


def fit(
    X,
    k_count,
    n_iter=100,
    tol=1e-6,
    reg_covar=1e-6,
    rng=None,
    init="random",
    init_params=None,
    clamped_labels=None,
):
    """Fit a K-component GMM to X by EM.

    X: (N, D). Returns a GMMFit. `rng`: a numpy Generator for reproducible
    initialization (pass the same seed to get the same run back); a fresh one
    is created if omitted. `init`: "random" (default) or "kmeans++" (see
    _init_random / _init_kmeanspp above for what each does and why). Raises
    NonMonotonicLikelihoodError (see above) if the log-likelihood ever
    decreases by more than floating-point noise, which by Section 2.6's proof is
    never a modeling issue and always a bug, UNLESS `clamped_labels` is given
    (see below), since that guarantee was proven for the standard,
    fully-unlabeled E-step specifically and doesn't automatically carry over
    once some responsibilities are fixed rather than recomputed by Bayes'
    rule.

    init_params: optional (pi, mu, Sigma) to start from directly, skipping
    `init` entirely. Used to WARM-START EM from a previous fit's parameters
    (RQ3: after a new label is revealed mid-query-loop, continue from the
    existing model instead of re-initializing from scratch).

    clamped_labels: optional (N,) int array; clamped_labels[n] = k means point
    n's true label is known to be component k, so its responsibility is fixed
    to a one-hot vector at k for every E-step, never recomputed. -1 means
    "not clamped," treated normally. See lessons/04, Section 4.1.
    """
    init_indices = None
    if init_params is not None:
        pi, mu, Sigma = init_params
    else:
        if init not in _INIT_STRATEGIES:
            raise ValueError(
                f"init must be one of {list(_INIT_STRATEGIES)}, got {init!r}"
            )
        rng = rng if rng is not None else np.random.default_rng()
        pi, mu, Sigma, init_indices = _INIT_STRATEGIES[init](X, k_count, reg_covar, rng)

    ll_history = [log_likelihood(X, pi, mu, Sigma)]
    converged = False

    for iteration in range(n_iter):
        r = responsibilities(X, pi, mu, Sigma)  # E-step
        if clamped_labels is not None:
            r = _apply_clamping(r, clamped_labels)
        pi, mu, Sigma = m_step(X, r, reg_covar=reg_covar)  # M-step

        ll = log_likelihood(X, pi, mu, Sigma)
        ll_history.append(ll)
        improvement = ll - ll_history[-2]

        if clamped_labels is None and improvement < -1e-8:
            # a REAL decrease, with no clamping to explain it, never "just
            # converged"
            raise NonMonotonicLikelihoodError(
                f"log-likelihood decreased by {-improvement:.3e} at iteration "
                f"{iteration} (from {ll_history[-2]:.6f} to {ll:.6f}). This "
                "should be impossible (lessons/02, Section 2.6) -- an E-step or "
                "M-step bug, not a modeling issue."
            )
        # abs(), not just `improvement < tol`: in the unclamped path this
        # changes nothing (the raise above already guarantees improvement
        # >= -1e-8 by the time we get here). In the clamped path, a real,
        # large drop is possible and must NOT be mistaken for convergence.
        # `improvement < tol` alone would call a drop of -5.0 "converged"
        # (since -5.0 < 1e-6), exactly the kind of mistake
        # NonMonotonicLikelihoodError exists to catch in the unclamped case.
        if abs(improvement) < tol:
            converged = True
            break

    return GMMFit(
        pi=pi,
        mu=mu,
        Sigma=Sigma,
        log_likelihood_history=ll_history,
        converged=converged,
        init_indices=init_indices,
    )


def fit_many(X, k_count, n_restarts=10, rng=None, **fit_kwargs):
    """Run fit() n_restarts times from different random initializations.

    Returns results from however many restarts succeeded (usually all
    n_restarts, occasionally fewer, see below), rather than just the best.
    This is the piece fit_best_of (below) is built from, and what RQ1/RQ2
    need directly: the SPREAD of outcomes across restarts (e.g. variance of
    the final log-likelihood) is itself the quantity of interest when
    studying initialization sensitivity, not only the single best answer.

    rng: a numpy Generator used to generate n_restarts independent sub-seeds
    (so the whole call is reproducible from one seed). Extra keyword
    arguments (n_iter, tol, reg_covar) are passed through to every fit().

    A restart that raises NonMonotonicLikelihoodError is skipped, not
    propagated. Verified three times independently (twice in
    research_log/RQ5c_outliers.md, once while running RQ6) that this
    specific error can come from genuine floating-point precision loss,
    rather than an E-step/M-step bug, when a restart's covariance approaches
    the near-singular collapse lessons/02 already describes (the error
    disappears without the restart's particular random seed; another
    restart on the same data converges fine). fit_many exists precisely so
    one unlucky restart doesn't derail the whole ensemble. is_degenerate
    (below) already filters out restarts that converge to a collapsed
    answer; this extends the same robustness to a restart that fails to
    converge at all. fit()'s own check is untouched and still raises
    immediately on a single, non-ensemble fit, since that's what actually
    catches a real bug. Raises RuntimeError if every restart fails, since
    returning nothing would be worse than being loud about it.
    """
    rng = rng if rng is not None else np.random.default_rng()
    seeds = rng.integers(0, 2**32 - 1, size=n_restarts)
    results = []
    for s in seeds:
        try:
            results.append(fit(X, k_count, rng=np.random.default_rng(s), **fit_kwargs))
        except NonMonotonicLikelihoodError:
            continue
    if not results:
        raise RuntimeError(
            f"all {n_restarts} restarts failed with NonMonotonicLikelihoodError -- "
            "this is no longer just one unlucky seed; investigate rather than retry."
        )
    return results


def is_degenerate(result, min_weight=0.02):
    """Flag a fit that collapsed instead of finding real cluster structure.

    lessons/02_jensen_and_elbo.md, Section 2.6 already warns that a full-covariance
    GMM's log-likelihood is UNBOUNDED ABOVE: one component can shrink onto a
    near-singular sliver of just a few points and claim near-infinite density
    there, beating any honest fit on raw log-likelihood. `reg_covar` (mstep.py)
    stops it from reaching literal infinity, but not from getting close enough
    to still win a "pick the highest log-likelihood" comparison. Found
    directly while building RQ1 (research_log/IDEAS.md has the exact
    numbers): at c=2, letting EM fully converge finds pi=[0.007, 0.993], one
    component down to 3 of 600 points with a covariance eigenvalue of 2e-5,
    which beats every honest 2-cluster solution on log-likelihood alone.

    A component capturing a near-vanishing share of the data isn't really a
    cluster; `min_weight` is the threshold below which we call that collapse
    rather than genuine structure. 0.02 (2%) is a judgement call, not a
    derived number: on K=2, "a real cluster" should usually hold much more
    than 2% of the data, and 2e-5-scale eigenvalues are nowhere near that
    boundary, so this threshold isn't finely tuned to the one case that
    found it.
    """
    return result.pi.min() < min_weight


def best_of(results, min_weight=0.02):
    """Pick the best ALREADY-COMPUTED fit from a list (e.g. from fit_many),
    without re-running anything: highest final log-likelihood among the
    non-degenerate ones (is_degenerate, above), falling back to the overall
    best only if every result collapsed, since returning nothing would be
    worse. fit_best_of (below) is `best_of(fit_many(...))`, nothing more.
    It's pulled out as its own function so code that already has a
    fit_many() result (e.g. experiments/rq2_initialization.py, which also
    needs the full spread of results, not just the winner) doesn't have to
    call fit_many a second time just to get the winner.
    """
    healthy = [r for r in results if not is_degenerate(r, min_weight)]
    pool = healthy if healthy else results
    return max(pool, key=lambda result: result.log_likelihood_history[-1])


def fit_best_of(X, k_count, n_restarts=10, rng=None, min_weight=0.02, **fit_kwargs):
    """Run fit() n_restarts times from different random initializations, keep
    the one with the highest final log-likelihood AMONG THE NON-DEGENERATE ones
    (is_degenerate, above), falling back to the overall best only if every
    restart collapsed, since returning nothing would be worse.

    A single EM run only finds A local optimum (lessons/01_latent_variables_and_gmm.md,
    Section 1.6), which one depends on where it started (lessons/01, Section 1.7 / the
    initialization-sensitivity finding recorded in research_log/IDEAS.md).
    Running several different random starts and keeping the best is the
    standard PRACTICAL mitigation, not a fix for that sensitivity; RQ2 is
    where we study HOW OFTEN and WHY it happens, systematically, across a
    controlled grid of conditions. This is just the everyday tool, used here
    (e.g. in experiments/stage4_sanity_check.py) so a single unlucky seed
    doesn't get mistaken for a correctness bug in a sanity check.

    If you already have a fit_many() result, call best_of() on it directly.
    This function exists for the common case of wanting only the winner.
    """
    candidates = fit_many(X, k_count, n_restarts=n_restarts, rng=rng, **fit_kwargs)
    return best_of(candidates, min_weight)
