"""Stage 4 sanity check: run this yourself, don't just read it.

    .venv/bin/python experiments/stage4_sanity_check.py

What it does, in order:
  1. Makes an EASY synthetic dataset: 2 obvious, well-separated clusters.
     Fits gmmlab (a single EM run) and scikit-learn to it, prints both.
  2. Makes a HARDER dataset: 3 overlapping clusters, unequal shapes. Fits
     BOTH gmmlab and scikit-learn to it too (see "Why best-of-N restarts
     for the harder case but not the easy one?" below).
  3. Saves one figure, figures/stage4_sanity_check.png, with 6 panels: for
     each case, the fit itself, its log-likelihood trace, and gmmlab vs
     sklearn overlaid.

Why best-of-N restarts for the harder case but not the easy one?
  A single EM run only finds A local optimum (lessons/01, Section 1.6), and which
  one depends on initialization (lessons/01, Section 1.7). On the easy, well-
  separated data, a single run reliably finds the right answer; the 41
  tests in tests/ confirm this across every function gmmlab is built from.
  On the HARDER, overlapping data, a single unlucky seed can land in a
  clearly worse optimum (verified: one specific seed here reaches a
  log-likelihood ~96 points worse than a good one, see
  tests/test_em.py::test_single_restart_can_land_in_a_worse_local_optimum,
  and research_log/IDEAS.md for a second, independent instance of the same
  thing). That is NOT a correctness bug: every one of the 41 tests still
  passes. It's initialization sensitivity, exactly what RQ2 exists to
  study properly. Using gmmlab.em.fit_best_of here (10 restarts, keep the
  best) is the practical fix so this sanity check isn't at the mercy of one
  unlucky draw; it is not RQ2's systematic investigation of the phenomenon
  itself.

Read lessons/matplotlib_for_this_project.md alongside this file, it
explains every plotting call used here, the same way numpy_for_this_project.md
explains the NumPy calls in gmmlab itself.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture

from gmmlab.em import fit, fit_best_of
from gmmlab.matching import match_components
from gmmlab.plotting import covariance_ellipse


def make_easy_dataset(rng, n_per_component=150):
    """2 components, well separated, equal spherical covariance."""
    true_mu = np.array([[-6.0, 0.0], [6.0, 0.0]])
    true_cov = np.eye(2)
    labels = np.repeat([0, 1], n_per_component)
    x = np.vstack(
        [true_mu[k] + rng.normal(size=(n_per_component, 2)) @ true_cov for k in (0, 1)]
    )
    return x, labels, true_mu


def make_hard_dataset(rng, n_per_component=150):
    """3 components, overlapping, unequal (non-spherical) covariance --
    exercises the FULL covariance fit (see lessons/03, Section 3.6's decision)."""
    true_mu = np.array([[-2.0, 0.0], [2.0, 1.5], [0.0, -2.5]])
    true_covs = [
        np.array([[2.0, 0.8], [0.8, 0.5]]),
        np.array([[0.6, -0.3], [-0.3, 1.5]]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    ]
    labels = np.repeat([0, 1, 2], n_per_component)
    chunks = []
    for k in range(3):
        l_factor = np.linalg.cholesky(true_covs[k])
        chunks.append(true_mu[k] + rng.normal(size=(n_per_component, 2)) @ l_factor.T)
    return np.vstack(chunks), labels, true_mu


def print_comparison(name, true_mu, ours, sk, x):
    perm = match_components(ours.mu, sk.means_)
    print(f"=== {name} ===")
    print("true means:         ", true_mu.tolist())
    print("gmmlab recovered:    ", ours.mu.round(3).tolist())
    print("sklearn recovered:   ", sk.means_[perm].round(3).tolist())
    print(f"gmmlab log-likelihood: {ours.log_likelihood_history[-1]:.4f}")
    print(f"sklearn log-likelihood: {sk.score(x) * len(x):.4f}")
    print(f"gmmlab converged in {ours.n_iter} EM iterations")
    diffs = np.diff(ours.log_likelihood_history)
    print(f"log-likelihood ever decreased? {(diffs < -1e-8).any()}  (should be False)")
    return perm


def draw_fit_panel(ax, x, labels, cmap, ours, title):
    k_count = len(ours.pi)
    ax.scatter(x[:, 0], x[:, 1], c=labels, cmap=cmap, s=10, alpha=0.5)
    ax.scatter(
        ours.mu[:, 0], ours.mu[:, 1], c="black", marker="x", s=150, label="gmmlab mean"
    )
    for k in range(k_count):
        covariance_ellipse(
            ax, ours.mu[k], ours.Sigma[k], edgecolor="black", linewidth=2
        )
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")


def draw_ll_panel(ax, ours, title):
    ax.plot(ours.log_likelihood_history, marker="o")
    ax.set_title(title)
    ax.set_xlabel("iteration")
    ax.set_ylabel("L(theta)")
    ax.grid(alpha=0.3)


def draw_overlay_panel(ax, x, ours, sk, perm, k_count, title):
    # gmmlab: thick, translucent "band", not meant to be read as a precise
    # line, just a wide highlight. sklearn: thin, fully opaque line drawn ON
    # TOP. If the two fits agree, the thin line sits visibly centred inside
    # the band; if they disagreed, the thin line would poke outside it.
    # Solid-vs-dashed at equal width made the two blend together exactly
    # where they overlap, which is the one place this plot most needs to be
    # legible.
    ax.scatter(x[:, 0], x[:, 1], c="lightgray", s=10, alpha=0.5)
    for k in range(k_count):
        covariance_ellipse(
            ax, ours.mu[k], ours.Sigma[k], edgecolor="C0", linewidth=8, alpha=0.35
        )
        covariance_ellipse(
            ax,
            sk.means_[perm[k]],
            sk.covariances_[perm[k]],
            edgecolor="C1",
            linewidth=1.5,
        )
    ax.plot([], [], color="C0", linewidth=8, alpha=0.35, label="gmmlab (band)")
    ax.plot([], [], color="C1", linewidth=1.5, label="sklearn (line)")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")


def main():
    rng = np.random.default_rng(0)

    # ---- easy case: a single EM run is already reliable here ----
    x_easy, labels_easy, true_mu_easy = make_easy_dataset(rng)
    ours_easy = fit(x_easy, 2, rng=np.random.default_rng(1))
    sk_easy = GaussianMixture(n_components=2, random_state=1, reg_covar=1e-6).fit(
        x_easy
    )
    perm_easy = print_comparison(
        "Easy case (2 obvious clusters)", true_mu_easy, ours_easy, sk_easy, x_easy
    )

    # ---- harder case: best-of-10 restarts (see module docstring for why) ----
    x_hard, labels_hard, true_mu_hard = make_hard_dataset(rng)
    ours_hard = fit_best_of(x_hard, 3, n_restarts=10, rng=np.random.default_rng(99))
    sk_hard = GaussianMixture(n_components=3, random_state=2, reg_covar=1e-6).fit(
        x_hard
    )
    print()
    perm_hard = print_comparison(
        "Harder case (3 overlapping, differently-shaped clusters)",
        true_mu_hard,
        ours_hard,
        sk_hard,
        x_hard,
    )

    # ---- the figure: one row per case, 3 panels each ----
    fig, axes = plt.subplots(3, 2, figsize=(11, 15))

    draw_fit_panel(
        axes[0, 0],
        x_easy,
        labels_easy,
        "coolwarm",
        ours_easy,
        "Easy case: data + our fit",
    )
    draw_ll_panel(axes[0, 1], ours_easy, "Easy case: log-likelihood per EM iteration")

    draw_fit_panel(
        axes[1, 0],
        x_hard,
        labels_hard,
        "viridis",
        ours_hard,
        "Harder case: best-of-10 fit",
    )
    draw_ll_panel(
        axes[1, 1], ours_hard, "Harder case: log-likelihood (winning restart)"
    )

    draw_overlay_panel(
        axes[2, 0],
        x_easy,
        ours_easy,
        sk_easy,
        perm_easy,
        2,
        "Easy case: gmmlab vs sklearn",
    )
    draw_overlay_panel(
        axes[2, 1],
        x_hard,
        ours_hard,
        sk_hard,
        perm_hard,
        3,
        "Harder case: gmmlab vs sklearn",
    )

    fig.tight_layout()
    out_path = "figures/stage4_sanity_check.png"
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
