"""RQ6c: do bootstrap ensembles provide information beyond responsibility
entropy? This is the other half of RQ6's locked question: it varies the
DATA (via resampling with replacement) instead of the starting point.

    .venv/bin/python experiments/rq6c_bootstrap_disagreement.py

Design is frozen in research_log/RQ6c_bootstrap_disagreement.md, so read
that alongside this file. Reuses RQ6's query-loop structure and its
entropy tie-break fix as-is, swapping in bootstrap resampling in place of
fresh random restarts for the ensemble.

Produces: a printed win-rate summary against the pre-committed hypothesis,
and figures/rq6c_bootstrap_disagreement.png.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import experiments.rq3_entropy_vs_random as rq3  # noqa: E402
from gmmlab.datasets import make_two_gaussians  # noqa: E402
from gmmlab.em import fit, fit_best_of  # noqa: E402
from gmmlab.matching import match_components  # noqa: E402
from gmmlab.responsibilities import responsibilities  # noqa: E402

N_ENSEMBLE = 5
ENSEMBLE_N_ITER = 500
N_TRIALS = 30


def bootstrap_disagreement_scores(x, clamped_labels, main_mu, rng):
    """5 bootstrap resamples (with replacement, same size), fit fresh on
    each, evaluated back on the ORIGINAL points (not the resample) so every
    candidate gets a vote from every ensemble member. Realigned to main_mu's
    labeling before comparing, same necessity as restart-disagreement."""
    n = len(x)
    votes = np.empty((N_ENSEMBLE, n), dtype=int)
    for i in range(N_ENSEMBLE):
        boot_idx = rng.integers(0, n, size=n)
        x_boot = x[boot_idx]
        clamped_boot = clamped_labels[boot_idx]
        seed = int(rng.integers(0, 2**32 - 1))
        result = fit(
            x_boot,
            2,
            n_iter=ENSEMBLE_N_ITER,
            rng=np.random.default_rng(seed),
            init="random",
            clamped_labels=clamped_boot,
        )
        perm = match_components(result.mu, main_mu)
        r = responsibilities(
            x, result.pi, result.mu, result.Sigma
        )  # evaluate on ORIGINAL x
        pred_raw = np.argmax(r, axis=1)
        votes[i] = perm[pred_raw]

    disagreement = np.empty(n)
    for j in range(n):
        counts = np.bincount(votes[:, j], minlength=2)
        disagreement[j] = 1.0 - counts.max() / N_ENSEMBLE
    return disagreement


def run_query_loop(x, z_true, perm0, pi0, mu0, Sigma0, strategy, rng):
    n = len(x)
    clamped_labels = np.full(n, -1)
    pi, mu, Sigma = pi0.copy(), mu0.copy(), Sigma0.copy()
    inverse_perm = np.argsort(perm0)
    accuracies = np.empty(rq3.QUERY_BUDGET)

    for step in range(rq3.QUERY_BUDGET):
        unlabeled = np.where(clamped_labels == -1)[0]

        if strategy == "entropy":
            r = responsibilities(x, pi, mu, Sigma)
            h = rq3.entropy(r)
            next_idx = unlabeled[np.argmax(h[unlabeled])]
        else:  # "bootstrap_disagreement"
            disagreement = bootstrap_disagreement_scores(x, clamped_labels, mu, rng)
            d_unlabeled = disagreement[unlabeled]
            tied = unlabeled[d_unlabeled == d_unlabeled.max()]
            if len(tied) > 1:
                r = responsibilities(x, pi, mu, Sigma)
                h = rq3.entropy(r)
                next_idx = tied[np.argmax(h[tied])]
            else:
                next_idx = tied[0]

        clamped_labels[next_idx] = int(inverse_perm[z_true[next_idx]])

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
        pred = perm0[np.argmax(r_final, axis=1)]
        accuracies[step] = (
            (pred[remaining] == z_true[remaining]).mean() if len(remaining) else 1.0
        )

    return accuracies


def run_trial(trial_idx):
    data_seed = trial_idx + 120_000
    restart_seed = trial_idx + 130_000
    ensemble_seed = (
        trial_idx + 150_000
    )  # distinct from RQ6's 140_000 restart-ensemble seed

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

    entropy_curve = run_query_loop(
        x, z_true, perm0, initial.pi, initial.mu, initial.Sigma, "entropy", None
    )
    disagreement_curve = run_query_loop(
        x,
        z_true,
        perm0,
        initial.pi,
        initial.mu,
        initial.Sigma,
        "bootstrap_disagreement",
        np.random.default_rng(ensemble_seed),
    )

    entropy_final = entropy_curve[-1]
    disagreement_final = disagreement_curve[-1]

    return {
        "trial": trial_idx,
        "entropy_curve": entropy_curve,
        "disagreement_curve": disagreement_curve,
        "entropy_final": entropy_final,
        "disagreement_final": disagreement_final,
        "win": disagreement_final
        > entropy_final,  # "win" = bootstrap-disagreement beats entropy
    }


def main():
    rows = [run_trial(t) for t in range(N_TRIALS)]

    n_wins = sum(r["win"] for r in rows)
    win_rate = n_wins / N_TRIALS
    print(
        f"Bootstrap-disagreement beat entropy in {n_wins}/{N_TRIALS} "
        f"= {win_rate:.1%} of trials"
    )
    print("For comparison, RQ6's restart-disagreement won 56.7% against entropy.")
    print("Predicted: not exceed 50%.  'Wrong' line: exceeds 50%.")
    verdict = (
        "CONSISTENT with the hypothesis"
        if win_rate <= 0.50
        else "HYPOTHESIS WRONG (exceeds the 50% line committed in advance)"
    )
    print(f"Verdict: {verdict}")

    entropy_all = np.array([r["entropy_curve"] for r in rows])
    disagreement_all = np.array([r["disagreement_curve"] for r in rows])
    queries = np.arange(1, rq3.QUERY_BUDGET + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, data, color in [
        ("entropy sampling", entropy_all, "C0"),
        ("bootstrap-disagreement sampling", disagreement_all, "C5"),
    ]:
        mean = data.mean(axis=0)
        lo = np.percentile(data, 25, axis=0)
        hi = np.percentile(data, 75, axis=0)
        ax.plot(queries, mean, label=label, color=color)
        ax.fill_between(queries, lo, hi, alpha=0.2, color=color)
    ax.set_xlabel("queries asked")
    ax.set_ylabel("held-out accuracy (still-unlabeled points)")
    ax.set_title(
        f"RQ6c: bootstrap-disagreement vs. entropy (c={rq3.C}, {N_TRIALS} trials)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = "figures/rq6c_bootstrap_disagreement.png"
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
