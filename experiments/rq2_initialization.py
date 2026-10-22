"""RQ2: how sensitive is EM to initialization, and how does that interact
with overlap?

    .venv/bin/python experiments/rq2_initialization.py

Read research_log/RQ2_initialization.md, Sections 3-6, alongside this file. Every
choice below (the c/trial grid, N, R, n_iter, the two init strategies, the
thresholds) is explained and justified there, not here; this file is just
that design turned into code.

Produces: a printed summary table, and figures/rq2_initialization.png (one
panel per c: the distribution of the winning fit's smallest component weight,
across all trials, random vs. kmeans++ init, side by side).
"""

import matplotlib.pyplot as plt
import numpy as np

from gmmlab.datasets import make_two_gaussians
from gmmlab.em import best_of, fit_many

C_CONFIG = [(0.5, 30), (1.0, 30), (4.0, 10)]  # (c, n_trials); see Section 3
SIGMA = 1.0
N_PER_COMPONENT = 300
N_RESTARTS = 10
N_ITER = 2000
LOPSIDED_THRESHOLD = 0.10  # generous reading of "lopsided" (Section 5)
DEGENERATE_THRESHOLD = 0.02  # matches gmmlab.em.is_degenerate's default


def run_trial(c, trial_idx, init):
    """One independent trial: fresh data, fresh restarts, one init strategy.
    Returns the winning fit's smallest component weight (min_k pi_k)."""
    data_seed = trial_idx
    restart_seed = trial_idx + 10_000  # offset so data/restart seeds never collide
    x, _, _ = make_two_gaussians(
        c,
        sigma=SIGMA,
        n_per_component=N_PER_COMPONENT,
        rng=np.random.default_rng(data_seed),
    )
    results = fit_many(
        x,
        2,
        n_restarts=N_RESTARTS,
        n_iter=N_ITER,
        rng=np.random.default_rng(restart_seed),
        init=init,
    )
    winner = best_of(results)
    return winner.pi.min()


def main():
    rows = []
    for c, n_trials in C_CONFIG:
        for init in ("random", "kmeans++"):
            min_weights = np.array([run_trial(c, t, init) for t in range(n_trials)])
            n_lopsided = int((min_weights < LOPSIDED_THRESHOLD).sum())
            n_degenerate = int((min_weights < DEGENERATE_THRESHOLD).sum())
            rows.append(
                {
                    "c": c,
                    "init": init,
                    "n_trials": n_trials,
                    "min_weights": min_weights,
                    "median": float(np.median(min_weights)),
                    "n_lopsided": n_lopsided,
                    "n_degenerate": n_degenerate,
                }
            )
            print(
                f"c={c:>4} init={init:10s} n={n_trials:2d}  "
                f"median min-weight={np.median(min_weights):.4f}  "
                f"<0.10: {n_lopsided:2d}/{n_trials}  "
                f"<0.02: {n_degenerate:2d}/{n_trials}"
            )

    n_panels = len(C_CONFIG)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(4.5 * n_panels + 1.5, 5), sharey=True
    )
    axes = np.atleast_1d(
        axes
    )  # plt.subplots(1, 1, ...) returns a bare Axes, not an array
    rng_jitter = np.random.default_rng(0)

    for ax, (c, _n_trials) in zip(axes, C_CONFIG, strict=True):
        c_rows = [r for r in rows if r["c"] == c]
        for i, r in enumerate(c_rows):
            x_jitter = i + rng_jitter.uniform(-0.15, 0.15, size=len(r["min_weights"]))
            ax.scatter(x_jitter, r["min_weights"], alpha=0.6, s=25)
            ax.scatter([i], [r["median"]], color="black", marker="_", s=400, zorder=5)
        ax.axhline(
            LOPSIDED_THRESHOLD, color="orange", linestyle=":", label="lopsided (<0.10)"
        )
        ax.axhline(
            DEGENERATE_THRESHOLD, color="red", linestyle=":", label="degenerate (<0.02)"
        )
        ax.axhline(0.5, color="green", linestyle="--", label="perfectly balanced")
        ax.set_xticks(range(len(c_rows)))
        ax.set_xticklabels([r["init"] for r in c_rows])
        ax.set_title(f"c = {c}")
        ax.set_ylim(-0.02, 0.55)

    axes[0].set_ylabel("winning fit's smallest component weight")
    axes[0].legend(fontsize=8, loc="center left")
    fig.suptitle(
        "Each dot = one independent trial. Black tick = median. "
        "Low = lopsided/confident-but-wrong (RQ1); near 0.5 = balanced."
    )
    fig.tight_layout()
    out_path = "figures/rq2_initialization.png"
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
