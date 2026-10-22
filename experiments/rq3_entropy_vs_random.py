"""RQ3: under what conditions does responsibility entropy improve learning
efficiency relative to random selection when labels are limited?

    .venv/bin/python experiments/rq3_entropy_vs_random.py

The design decisions here (c, the query budget, the metric, warm-starting,
the repeat structure) all live in research_log/RQ3_entropy_vs_random.md,
not in this file, so read that first. This is just the design written out
as code.

Produces: a printed win-rate summary against the pre-committed hypothesis,
and figures/rq3_entropy_vs_random.png (accuracy-vs-queries, both strategies,
mean +/- interquartile band across trials).
"""

import matplotlib.pyplot as plt
import numpy as np

from gmmlab.datasets import make_two_gaussians
from gmmlab.em import fit, fit_best_of
from gmmlab.matching import match_components
from gmmlab.responsibilities import responsibilities

C = 1.0
SIGMA = 1.0
N_PER_COMPONENT = 300  # N=600, matching RQ1/RQ2
N_RESTARTS = 10  # for the one initial fit per trial, matching RQ1/RQ2
INIT_N_ITER = 2000  # for the initial fit only, since overlapping data converges slowly
WARM_N_ITER = (
    200  # per incremental refit after a reveal; warm-started, so it converges fast
)
QUERY_BUDGET = 60
N_TRIALS = 30  # independent datasets
N_RANDOM_SUBREPEATS = 5  # random's own choices are randomized; uncertainty's aren't
LOPSIDED_THRESHOLD = 0.10  # same rule as RQ2, used for the post-hoc breakdown only,
# not part of the pre-registered hypothesis (research_log/RQ3_entropy_vs_random.md)


def entropy(r):
    r = np.clip(r, 1e-300, 1.0)
    return -np.sum(r * np.log(r), axis=1)


def run_query_loop(x, z_true, perm, pi0, mu0, Sigma0, strategy, rng):
    """QUERY_BUDGET reveal-and-warm-refit steps. `perm` (from
    match_components(initial_mu, mu_true), computed ONCE per trial before any
    query) is the fixed correspondence between the model's component indices
    and the data generator's true labels, needed because a point's true
    label (0 or 1) is not automatically the model's own component index for
    that cluster (that correspondence is arbitrary, Lesson/RQ1's
    label-switching point). Returns held-out accuracy after each step."""
    # match_components(fitted_mu, mu_true) returns perm with perm[fitted_k] =
    # the TRUE label fitted component k corresponds to (matching.py's
    # docstring: mu_true[perm] lines up with fitted_mu component by
    # component). So perm converts fitted-index -> true-label (what accuracy
    # needs); its inverse converts true-label -> fitted-index (what clamping
    # needs, since clamping has to fix a RAW component index).
    n = len(x)
    clamped_labels = np.full(n, -1)
    pi, mu, Sigma = pi0.copy(), mu0.copy(), Sigma0.copy()
    inverse_perm = np.argsort(perm)
    accuracies = np.empty(QUERY_BUDGET)

    for step in range(QUERY_BUDGET):
        unlabeled = np.where(clamped_labels == -1)[0]

        if strategy == "uncertainty":
            r = responsibilities(x, pi, mu, Sigma)
            h = entropy(r)
            next_idx = unlabeled[np.argmax(h[unlabeled])]
        elif strategy == "oracle_wrong":
            # RQ5: a deliberately unrealistic diagnostic, checks against
            # z_true (only possible because this is synthetic data) to
            # target whatever the model is currently most
            # CONFIDENT about and WRONG about, isolating the Lesson 4, Section 4.4
            # blind spot from the general redundancy problem (RQ3, Section 9).
            r = responsibilities(x, pi, mu, Sigma)
            h = entropy(r)
            pred_now = perm[np.argmax(r, axis=1)]
            wrong_unlabeled = unlabeled[pred_now[unlabeled] != z_true[unlabeled]]
            if len(wrong_unlabeled) > 0:
                next_idx = wrong_unlabeled[np.argmin(h[wrong_unlabeled])]
            else:  # no wrong points left, fall back to most uncertain
                next_idx = unlabeled[np.argmax(h[unlabeled])]
        else:  # "random"
            next_idx = unlabeled[rng.integers(len(unlabeled))]

        # true label -> the RAW fitted component index that currently
        # represents it
        clamped_labels[next_idx] = int(inverse_perm[z_true[next_idx]])

        result = fit(
            x,
            2,
            n_iter=WARM_N_ITER,
            init_params=(pi, mu, Sigma),
            clamped_labels=clamped_labels,
        )
        pi, mu, Sigma = result.pi, result.mu, result.Sigma

        remaining = np.where(clamped_labels == -1)[0]
        r_final = responsibilities(x, pi, mu, Sigma)
        # raw fitted component index -> true label
        pred = perm[np.argmax(r_final, axis=1)]
        accuracies[step] = (
            (pred[remaining] == z_true[remaining]).mean() if len(remaining) else 1.0
        )

    return accuracies


def run_trial(trial_idx):
    data_seed = trial_idx
    restart_seed = trial_idx + 10_000
    random_strategy_seed = trial_idx + 20_000

    x, z_true, mu_true = make_two_gaussians(
        C,
        sigma=SIGMA,
        n_per_component=N_PER_COMPONENT,
        rng=np.random.default_rng(data_seed),
    )
    initial = fit_best_of(
        x,
        2,
        n_restarts=N_RESTARTS,
        n_iter=INIT_N_ITER,
        rng=np.random.default_rng(restart_seed),
    )
    perm = match_components(initial.mu, mu_true)

    uncertainty_curve = run_query_loop(
        x, z_true, perm, initial.pi, initial.mu, initial.Sigma, "uncertainty", None
    )

    rng_random = np.random.default_rng(random_strategy_seed)
    random_curves = np.array(
        [
            run_query_loop(
                x,
                z_true,
                perm,
                initial.pi,
                initial.mu,
                initial.Sigma,
                "random",
                rng_random,
            )
            for _ in range(N_RANDOM_SUBREPEATS)
        ]
    )

    uncertainty_final = uncertainty_curve[-1]
    random_final_mean = random_curves[:, -1].mean()

    return {
        "trial": trial_idx,
        "uncertainty_curve": uncertainty_curve,
        "random_curve_mean": random_curves.mean(axis=0),
        "uncertainty_final": uncertainty_final,
        "random_final_mean": random_final_mean,
        "win": uncertainty_final > random_final_mean,
        # post-hoc breakdown only (see LOPSIDED_THRESHOLD above), not part of
        # the pre-registered hypothesis itself
        "initial_lopsided": bool(initial.pi.min() < LOPSIDED_THRESHOLD),
    }


def main():
    rows = [run_trial(t) for t in range(N_TRIALS)]

    n_wins = sum(r["win"] for r in rows)
    win_rate = n_wins / N_TRIALS
    print(
        f"Uncertainty sampling beat random sampling in {n_wins}/{N_TRIALS} "
        f"= {win_rate:.1%} of trials"
    )
    print("Predicted: 60-70%.  'Wrong' line: below 50%.")
    verdict = (
        "CONSISTENT with the hypothesis"
        if win_rate >= 0.50
        else "HYPOTHESIS WRONG (below the 50% line committed in advance)"
    )
    print(f"Verdict: {verdict}")

    # post-hoc breakdown (not part of the hypothesis test above): does the
    # win rate depend on whether the INITIAL fit was already lopsided
    # (RQ1/RQ2's confident-but-wrong pattern), before any queries happened?
    for lopsided_flag, group_name in [
        (True, "started lopsided"),
        (False, "started balanced"),
    ]:
        group = [r for r in rows if r["initial_lopsided"] == lopsided_flag]
        if group:
            group_win_rate = sum(r["win"] for r in group) / len(group)
            print(
                f"  {group_name}: {sum(r['win'] for r in group)}/{len(group)} "
                f"= {group_win_rate:.1%} (n={len(group)})"
            )
        else:
            print(f"  {group_name}: no trials in this group")

    uncertainty_all = np.array([r["uncertainty_curve"] for r in rows])
    random_all = np.array([r["random_curve_mean"] for r in rows])
    queries = np.arange(1, QUERY_BUDGET + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, data, color in [
        ("uncertainty sampling", uncertainty_all, "C0"),
        ("random sampling", random_all, "C1"),
    ]:
        mean = data.mean(axis=0)
        lo = np.percentile(data, 25, axis=0)
        hi = np.percentile(data, 75, axis=0)
        ax.plot(queries, mean, label=label, color=color)
        ax.fill_between(queries, lo, hi, alpha=0.2, color=color)
    ax.set_xlabel("queries asked")
    ax.set_ylabel("held-out accuracy (still-unlabeled points)")
    ax.set_title(f"RQ3: uncertainty vs. random sampling (c={C}, {N_TRIALS} trials)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = "figures/rq3_entropy_vs_random.png"
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
