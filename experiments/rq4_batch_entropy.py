"""RQ4: does top-k entropy produce an informative batch, or a redundant one?

    .venv/bin/python experiments/rq4_batch_entropy.py

research_log/RQ4_batch_entropy.md documents the design decisions; read that
alongside this file. This script reuses experiments/rq3_entropy_vs_random.py's
`run_query_loop` as-is for the sequential condition (same function, not a
reimplementation) and its exact 30 trial seeds, so the two conditions stay
guaranteed-comparable instead of depending on two independently written
loops staying in sync.

Produces: a printed win-rate summary against the pre-committed hypothesis,
and figures/rq4_batch_entropy.png.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# running this file directly (`python experiments/rq4_batch_entropy.py`, as
# documented above) puts the script's OWN folder on the import path, not the
# project root, so `import experiments.rq3_...` can't find itself unless
# the root is added explicitly first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import experiments.rq3_entropy_vs_random as rq3  # noqa: E402
from gmmlab.datasets import make_two_gaussians
from gmmlab.em import fit, fit_best_of
from gmmlab.matching import match_components
from gmmlab.responsibilities import responsibilities

BATCH_SIZE = 10
N_ROUNDS = 6  # BATCH_SIZE * N_ROUNDS = 60 = rq3.QUERY_BUDGET


def run_batch(x, z_true, perm, pi0, mu0, Sigma0):
    """N_ROUNDS rounds of: rank all unlabeled points by entropy under the
    CURRENT parameters, reveal the top BATCH_SIZE simultaneously, refit once.
    See research_log/RQ4_batch_entropy.md, Section 3."""
    n = len(x)
    clamped_labels = np.full(n, -1)
    pi, mu, Sigma = pi0.copy(), mu0.copy(), Sigma0.copy()
    inverse_perm = np.argsort(perm)
    accuracies = np.empty(N_ROUNDS)

    for round_idx in range(N_ROUNDS):
        unlabeled = np.where(clamped_labels == -1)[0]
        r = responsibilities(x, pi, mu, Sigma)
        h = rq3.entropy(r)
        ranked = unlabeled[np.argsort(-h[unlabeled])]
        batch_idx = ranked[:BATCH_SIZE]

        clamped_labels[batch_idx] = inverse_perm[z_true[batch_idx]]

        result = fit(
            x,
            2,
            n_iter=rq3.WARM_N_ITER,
            init_params=(pi, mu, Sigma),
            clamped_labels=clamped_labels,
        )
        pi, mu, Sigma = result.pi, result.mu, result.Sigma

        remaining = np.where(clamped_labels == -1)[0]
        r_final = responsibilities(x, pi, mu, Sigma)
        pred = perm[np.argmax(r_final, axis=1)]
        accuracies[round_idx] = (
            (pred[remaining] == z_true[remaining]).mean() if len(remaining) else 1.0
        )

    return accuracies


def run_trial(trial_idx):
    # SAME seeds as RQ3, which guarantees the identical dataset and identical
    # initial fit, so the sequential condition reproduces RQ3's
    # "uncertainty sampling" curve exactly.
    data_seed = trial_idx
    restart_seed = trial_idx + 10_000

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
    perm = match_components(initial.mu, mu_true)

    sequential_curve = rq3.run_query_loop(
        x, z_true, perm, initial.pi, initial.mu, initial.Sigma, "uncertainty", None
    )
    batch_curve = run_batch(x, z_true, perm, initial.pi, initial.mu, initial.Sigma)

    sequential_final = sequential_curve[-1]
    batch_final = batch_curve[-1]

    return {
        "trial": trial_idx,
        "sequential_curve": sequential_curve,
        "batch_curve": batch_curve,
        "sequential_final": sequential_final,
        "batch_final": batch_final,
        "win": sequential_final > batch_final,  # "win" = sequential beats batch
    }


def main():
    rows = [run_trial(t) for t in range(rq3.N_TRIALS)]

    n_wins = sum(r["win"] for r in rows)
    win_rate = n_wins / rq3.N_TRIALS
    print(
        f"Sequential beat batch in {n_wins}/{rq3.N_TRIALS} = {win_rate:.1%} of trials"
    )
    print("Predicted: 50-60%.  'Wrong' line: below 40%.")
    verdict = (
        "CONSISTENT with the hypothesis"
        if win_rate >= 0.40
        else "HYPOTHESIS WRONG (below the 40% line committed in advance)"
    )
    print(f"Verdict: {verdict}")

    sequential_all = np.array([r["sequential_curve"] for r in rows])
    batch_all = np.array([r["batch_curve"] for r in rows])
    seq_queries = np.arange(1, rq3.QUERY_BUDGET + 1)
    batch_queries = np.arange(BATCH_SIZE, rq3.QUERY_BUDGET + 1, BATCH_SIZE)

    fig, ax = plt.subplots(figsize=(8, 5))
    seq_mean = sequential_all.mean(axis=0)
    ax.plot(seq_queries, seq_mean, label="sequential (1 at a time)", color="C0")
    ax.fill_between(
        seq_queries,
        np.percentile(sequential_all, 25, axis=0),
        np.percentile(sequential_all, 75, axis=0),
        alpha=0.2,
        color="C0",
    )
    batch_mean = batch_all.mean(axis=0)
    ax.plot(
        batch_queries,
        batch_mean,
        label="batch (top-10 at once)",
        color="C2",
        marker="o",
    )
    ax.fill_between(
        batch_queries,
        np.percentile(batch_all, 25, axis=0),
        np.percentile(batch_all, 75, axis=0),
        alpha=0.2,
        color="C2",
    )
    ax.set_xlabel("total queries used")
    ax.set_ylabel("held-out accuracy (still-unlabeled points)")
    ax.set_title(
        f"RQ4: sequential vs. batch entropy querying (c={rq3.C}, {rq3.N_TRIALS} trials)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = "figures/rq4_batch_entropy.png"
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
