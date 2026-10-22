"""RQ1: how does cluster overlap (separation c) affect EM's recovery of
latent structure and its assignment uncertainty?

    .venv/bin/python experiments/rq1_cluster_overlap.py

The design is frozen in research_log/RQ1_cluster_overlap.md, Section 3, so read
that alongside this file. Every choice below (the c grid, N, R, n_iter, the
degeneracy guard, the overlap-region definition) gets explained and
justified there, not here; this file is just that design turned into code.

Produces: a printed results table, and figures/rq1_cluster_overlap.png
(4 panels: accuracy vs Bayes-optimal, entropy decomposition, restart
log-likelihood variance, degenerate-restart fraction).
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from gmmlab.datasets import make_two_gaussians
from gmmlab.em import best_of, fit_many, is_degenerate
from gmmlab.matching import match_components
from gmmlab.plotting import covariance_ellipse
from gmmlab.responsibilities import responsibilities

C_GRID = [0.5, 1, 2, 3, 4, 6]
SIGMA = 1.0
N_PER_COMPONENT = 300
N_RESTARTS = 10
N_ITER = 2000
OVERLAP_HALF_WIDTH = 0.5 * SIGMA  # |x_n| <= this => "central overlap region"

DATA_SEED = 0  # SAME for every c: common random numbers (see Section 3)
RESTART_SEED = 1  # SAME for every c: common random numbers (see Section 3)


def entropy(r):
    """Row-wise entropy of a responsibility matrix (N, K) -> (N,)."""
    r = np.clip(r, 1e-300, 1.0)  # avoid log(0) for exact-0 responsibilities
    return -np.sum(r * np.log(r), axis=1)


def run_one_condition(c):
    x, z_true, mu_true = make_two_gaussians(
        c,
        sigma=SIGMA,
        n_per_component=N_PER_COMPONENT,
        rng=np.random.default_rng(DATA_SEED),
    )
    pi_true = np.array([0.5, 0.5])
    sigma_true = np.stack([SIGMA**2 * np.eye(2), SIGMA**2 * np.eye(2)])

    # Compute the 10 restarts ONCE and derive the winner from that same list
    # (best_of), instead of calling fit_many and fit_best_of separately,
    # since fit_best_of is best_of(fit_many(...)) internally, so calling both
    # with the same seed silently repeated all 10 restarts a second time.
    # Caught and fixed here; verified to change none of the already-
    # reported numbers in research_log/RQ1_cluster_overlap.md (same seed,
    # same computation, just done once instead of twice).
    all_restarts = fit_many(
        x,
        2,
        n_restarts=N_RESTARTS,
        n_iter=N_ITER,
        rng=np.random.default_rng(RESTART_SEED),
    )
    best = best_of(all_restarts)

    # --- label-recovery accuracy, matched against the truth ---
    perm = match_components(best.mu, mu_true)
    r_fitted = responsibilities(x, best.pi, best.mu, best.Sigma)
    pred_raw = np.argmax(r_fitted, axis=1)
    inverse_perm = np.argsort(perm)
    pred = inverse_perm[pred_raw]
    em_accuracy = (pred == z_true).mean()
    bayes_accuracy = 1 - norm.cdf(-c / 2)

    # --- entropy: global, and in the central overlap region, fitted vs true ---
    h_fitted = entropy(r_fitted)
    mask = np.abs(x[:, 0]) <= OVERLAP_HALF_WIDTH
    r_true = responsibilities(x, pi_true, mu_true, sigma_true)
    h_true = entropy(r_true)

    # --- restart spread: variance across the 10 restarts' final log-likelihoods ---
    restart_lls = np.array([r.log_likelihood_history[-1] for r in all_restarts])
    degenerate_fraction = np.mean([is_degenerate(r) for r in all_restarts])

    return {
        "c": c,
        "em_accuracy": em_accuracy,
        "bayes_accuracy": bayes_accuracy,
        "accuracy_gap": em_accuracy - bayes_accuracy,
        "entropy_all_fitted": h_fitted.mean(),
        "entropy_overlap_fitted": h_fitted[mask].mean(),
        "entropy_overlap_true": h_true[mask].mean(),
        "n_overlap_points": int(mask.sum()),
        "restart_ll_variance": restart_lls.var(),
        "restart_ll_range": restart_lls.max() - restart_lls.min(),
        "degenerate_fraction": degenerate_fraction,
        "best_converged": best.converged,
        # kept for the per-c picture (plot_fits_grid), not printed in the
        # results table, so this doesn't duplicate anything already reported
        "x": x,
        "z_true": z_true,
        "mu_true": mu_true,
        "best_fit": best,
    }


def plot_fits_grid(rows, out_path="figures/rq1_cluster_overlap_fits.png"):
    """One panel per c: the data (true colour), the true means, and what EM
    ended up recovering. Same visual language as Stage 4's sanity-check
    figure, so the overconfidence finding (research_log, Section 8) can be SEEN,
    not just read off the results table."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for ax, row in zip(axes.flat, rows, strict=True):
        x, z_true, mu_true, best = (
            row["x"],
            row["z_true"],
            row["mu_true"],
            row["best_fit"],
        )

        ax.scatter(x[:, 0], x[:, 1], c=z_true, cmap="coolwarm", s=8, alpha=0.4)
        ax.scatter(
            mu_true[:, 0],
            mu_true[:, 1],
            marker="D",
            c="black",
            s=70,
            label="true mean",
            zorder=5,
        )
        ax.scatter(
            best.mu[:, 0],
            best.mu[:, 1],
            marker="x",
            c="black",
            s=140,
            linewidths=2.5,
            label="EM's fitted mean",
            zorder=5,
        )
        for k in range(2):
            covariance_ellipse(
                ax, best.mu[k], best.Sigma[k], edgecolor="black", linewidth=1.5
            )
        # the overlap-region boundary used for H_ovl, drawn so it's visible
        # on the plot, not buried in a results-table number
        ax.axvline(-OVERLAP_HALF_WIDTH, color="gray", linestyle=":", linewidth=1)
        ax.axvline(OVERLAP_HALF_WIDTH, color="gray", linestyle=":", linewidth=1)

        weights = ", ".join(f"{p:.2f}" for p in sorted(best.pi))
        ax.set_title(f"c = {row['c']}   (fitted weights: {weights})")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal")
        if row is rows[0]:
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Diamonds = true means (what generated the data). "
        "X's + ellipses = what EM actually recovered. "
        "Dotted lines = the overlap region used for H_ovl.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"Saved {out_path}")


def main():
    rows = [run_one_condition(c) for c in C_GRID]

    header = (
        f"{'c':>5} {'EM acc':>8} {'Bayes acc':>10} {'gap':>8} "
        f"{'H_all':>8} {'H_ovl(fit)':>11} {'H_ovl(true)':>12} "
        f"{'n_ovl':>6} {'ll_var':>10} {'degen%':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['c']:>5.1f} {row['em_accuracy']:>8.4f} "
            f"{row['bayes_accuracy']:>10.4f} {row['accuracy_gap']:>8.4f} "
            f"{row['entropy_all_fitted']:>8.4f} {row['entropy_overlap_fitted']:>11.4f} "
            f"{row['entropy_overlap_true']:>12.4f} {row['n_overlap_points']:>6d} "
            f"{row['restart_ll_variance']:>10.3f} "
            f"{row['degenerate_fraction'] * 100:>6.1f}%"
        )

    c_vals = [r["c"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    ax.plot(c_vals, [r["em_accuracy"] for r in rows], marker="o", label="EM accuracy")
    ax.plot(
        c_vals,
        [r["bayes_accuracy"] for r in rows],
        marker="o",
        label="Bayes-optimal accuracy",
    )
    ax.set_xlabel("separation c")
    ax.set_ylabel("accuracy")
    ax.set_title("Label-recovery accuracy vs. the Bayes-optimal floor")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(
        c_vals,
        [r["entropy_all_fitted"] for r in rows],
        marker="o",
        label="all points (fitted)",
    )
    ax.plot(
        c_vals,
        [r["entropy_overlap_fitted"] for r in rows],
        marker="o",
        label="overlap region (fitted)",
    )
    ax.plot(
        c_vals,
        [r["entropy_overlap_true"] for r in rows],
        marker="o",
        linestyle="--",
        label="overlap region (TRUE params)",
    )
    ax.axhline(np.log(2), color="gray", linestyle=":", label="ln 2 (max for K=2)")
    ax.set_xlabel("separation c")
    ax.set_ylabel("mean entropy")
    ax.set_title("Assignment entropy: global vs. overlap region, fitted vs. true")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(c_vals, [r["restart_ll_variance"] for r in rows], marker="o", color="C3")
    ax.set_xlabel("separation c")
    ax.set_ylabel("variance of final log-likelihood across 10 restarts")
    ax.set_title("Restart spread (observation for RQ2)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(
        c_vals, [r["degenerate_fraction"] * 100 for r in rows], marker="o", color="C4"
    )
    ax.set_xlabel("separation c")
    ax.set_ylabel("% of 10 restarts flagged degenerate")
    ax.set_title("How often EM finds the collapse pathology (Section 3)")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path = "figures/rq1_cluster_overlap.png"
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved {out_path}")

    plot_fits_grid(rows)


if __name__ == "__main__":
    main()
