"""RQ6 follow-up: does restart-disagreement also escape RQ3's redundancy
problem (repeatedly querying near-duplicate points in one crowded,
regenerating boundary region), or does its RQ6 win come purely from the
leverage/overcorrection story?

    .venv/bin/python experiments/rq6b_redundancy_check.py

Design is frozen in research_log/RQ6_restart_disagreement.md, Section 12. This is
a fresh implementation (it imports restart_disagreement_scores from
rq6_restart_disagreement.py directly and doesn't touch that file's
already-published run_query_loop/run_trial), so there's zero regression
risk to RQ6's locked results.

Produces: a printed spread-ratio summary against the pre-committed
hypothesis, and figures/rq6b_redundancy_check.png.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import experiments.rq3_entropy_vs_random as rq3  # noqa: E402
import experiments.rq6_restart_disagreement as rq6  # noqa: E402
from gmmlab.datasets import make_two_gaussians  # noqa: E402
from gmmlab.em import fit, fit_best_of  # noqa: E402
from gmmlab.matching import match_components  # noqa: E402
from gmmlab.responsibilities import responsibilities  # noqa: E402

N_TRIALS = 30


def run_query_loop_tracked(x, z_true, perm0, pi0, mu0, Sigma0, strategy, rng):
    """Same mechanics as rq6.run_query_loop, but also records the index of
    every queried point, for the spread diagnostic."""
    n = len(x)
    clamped_labels = np.full(n, -1)
    pi, mu, Sigma = pi0.copy(), mu0.copy(), Sigma0.copy()
    inverse_perm = np.argsort(perm0)
    queried_indices = []

    for _step in range(rq3.QUERY_BUDGET):
        unlabeled = np.where(clamped_labels == -1)[0]

        if strategy == "entropy":
            r = responsibilities(x, pi, mu, Sigma)
            h = rq3.entropy(r)
            next_idx = unlabeled[np.argmax(h[unlabeled])]
        else:  # "restart_disagreement"
            disagreement = rq6.restart_disagreement_scores(x, clamped_labels, mu, rng)
            d_unlabeled = disagreement[unlabeled]
            tied = unlabeled[d_unlabeled == d_unlabeled.max()]
            if len(tied) > 1:
                r = responsibilities(x, pi, mu, Sigma)
                h = rq3.entropy(r)
                next_idx = tied[np.argmax(h[tied])]
            else:
                next_idx = tied[0]

        queried_indices.append(int(next_idx))
        clamped_labels[next_idx] = int(inverse_perm[z_true[next_idx]])

        result = fit(
            x,
            2,
            n_iter=rq3.WARM_N_ITER,
            init_params=(pi, mu, Sigma),
            clamped_labels=clamped_labels,
        )
        pi, mu, Sigma = result.pi, result.mu, result.Sigma

    return queried_indices


def run_trial(trial_idx):
    data_seed = trial_idx + 120_000
    restart_seed = trial_idx + 130_000
    ensemble_seed = trial_idx + 140_000

    x, z_true, mu_true = make_two_gaussians(
        rq3.C,
        sigma=rq3.SIGMA,
        n_per_component=rq3.N_PER_COMPONENT,
        rng=np.random.default_rng(data_seed),
    )
    initial = fit_best_of(
        x,
        2,
        n_restarts=rq3.N_RESTARTS,
        n_iter=rq3.INIT_N_ITER,
        rng=np.random.default_rng(restart_seed),
    )
    perm0 = match_components(initial.mu, mu_true)

    entropy_queried = run_query_loop_tracked(
        x, z_true, perm0, initial.pi, initial.mu, initial.Sigma, "entropy", None
    )
    disagreement_queried = run_query_loop_tracked(
        x,
        z_true,
        perm0,
        initial.pi,
        initial.mu,
        initial.Sigma,
        "restart_disagreement",
        np.random.default_rng(ensemble_seed),
    )

    entropy_x = x[entropy_queried, 0]
    disagreement_x = x[disagreement_queried, 0]
    entropy_spread = float(entropy_x.std())
    disagreement_spread = float(disagreement_x.std())

    return {
        "trial": trial_idx,
        "entropy_spread": entropy_spread,
        "disagreement_spread": disagreement_spread,
        "ratio": disagreement_spread / entropy_spread,
        "entropy_x": entropy_x,
        "disagreement_x": disagreement_x,
    }


def main():
    rows = [run_trial(t) for t in range(N_TRIALS)]

    ratios = np.array([r["ratio"] for r in rows])
    mean_ratio = ratios.mean()
    print(f"Mean spread ratio (restart-disagreement / entropy): {mean_ratio:.2f}x")
    print(
        f"Per-trial ratios: min={ratios.min():.2f}x max={ratios.max():.2f}x "
        f"median={np.median(ratios):.2f}x"
    )
    print("Predicted: 1.2x-2.0x.  'Wrong' line: below 1.2x.")
    verdict = (
        "CONSISTENT with the hypothesis"
        if mean_ratio >= 1.2
        else "HYPOTHESIS WRONG (below the 1.2x line committed in advance)"
    )
    print(f"Verdict: {verdict}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].hist(
        [r["entropy_spread"] for r in rows], bins=10, alpha=0.6, label="entropy"
    )
    axes[0].hist(
        [r["disagreement_spread"] for r in rows],
        bins=10,
        alpha=0.6,
        label="restart-disagreement",
    )
    axes[0].set_xlabel("spread (std of queried points' x-coordinate)")
    axes[0].set_ylabel("number of trials")
    axes[0].set_title("Query spread per trial, both strategies")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(ratios, bins=10, color="C4")
    axes[1].axvline(1.0, color="gray", linestyle=":", label="equal spread")
    axes[1].axvline(
        mean_ratio, color="C3", linestyle="--", label=f"mean ({mean_ratio:.2f}x)"
    )
    axes[1].set_xlabel("ratio (restart-disagreement spread / entropy spread)")
    axes[1].set_ylabel("number of trials")
    axes[1].set_title("Spread ratio per trial")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_path = "figures/rq6b_redundancy_check.png"
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
