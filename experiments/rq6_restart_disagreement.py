"""RQ6: do restart ensembles provide information beyond responsibility
entropy? This is the realistic, non-cheating version of RQ5a's oracle.
Instead of checking the true label, it uses disagreement across 5
independent restarts as a proxy for "this point might be confidently wrong."

    .venv/bin/python experiments/rq6_restart_disagreement.py

Design is frozen in research_log/RQ6_restart_disagreement.md, so read that
alongside this file. Reimplements RQ3's query loop because the new
strategy needs extra per-step machinery (5 fresh restarts, component-label
realignment via match_components) that doesn't fit cleanly into
rq3.run_query_loop's existing branches.

Produces: a printed win-rate summary against the pre-committed hypothesis,
and figures/rq6_restart_disagreement.png.
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


def restart_disagreement_scores(x, clamped_labels, main_mu, rng):
    """5 fresh restarts, constrained to respect clamped_labels, realigned
    to main_mu's component labeling before comparing (see the log's Section 3 for why
    this alignment step isn't optional). Returns a disagreement score in
    [0, 1] per point, the fraction of the 5 restarts that disagree with the
    majority vote for that point."""
    n = len(x)
    votes = np.empty((N_ENSEMBLE, n), dtype=int)
    for i in range(N_ENSEMBLE):
        seed = int(rng.integers(0, 2**32 - 1))
        result = fit(
            x,
            2,
            n_iter=ENSEMBLE_N_ITER,
            rng=np.random.default_rng(seed),
            init="random",
            clamped_labels=clamped_labels,
        )
        perm = match_components(result.mu, main_mu)
        r = responsibilities(x, result.pi, result.mu, result.Sigma)
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
        else:  # "restart_disagreement"
            disagreement = restart_disagreement_scores(x, clamped_labels, mu, rng)
            # with only N_ENSEMBLE+1 possible disagreement values across up
            # to 600 points, large exact ties are common (confirmed: 317/600
            # in one trial), so argmax would deterministically pick whichever
            # tied point happens to sit first in the array. That's just an
            # artifact of how make_two_gaussians lays out the data
            # (component 0's points first), not a real signal. Break ties
            # with entropy from the main fit instead: disagreement picks
            # the shortlist, entropy picks within it.
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
        "restart_disagreement",
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
        > entropy_final,  # "win" = restart-disagreement beats entropy
    }


def main():
    rows = [run_trial(t) for t in range(N_TRIALS)]

    n_wins = sum(r["win"] for r in rows)
    win_rate = n_wins / N_TRIALS
    print(
        f"Restart-disagreement beat entropy in {n_wins}/{N_TRIALS} "
        f"= {win_rate:.1%} of trials"
    )
    print("For comparison, RQ5a's oracle won 40.0% against entropy.")
    print("Predicted: 45-60%.  'Wrong' line: below 45%.")
    verdict = (
        "CONSISTENT with the hypothesis"
        if win_rate >= 0.45
        else "HYPOTHESIS WRONG (below the 45% line committed in advance)"
    )
    print(f"Verdict: {verdict}")

    entropy_all = np.array([r["entropy_curve"] for r in rows])
    disagreement_all = np.array([r["disagreement_curve"] for r in rows])
    queries = np.arange(1, rq3.QUERY_BUDGET + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, data, color in [
        ("entropy sampling", entropy_all, "C0"),
        ("restart-disagreement sampling", disagreement_all, "C4"),
    ]:
        mean = data.mean(axis=0)
        lo = np.percentile(data, 25, axis=0)
        hi = np.percentile(data, 75, axis=0)
        ax.plot(queries, mean, label=label, color=color)
        ax.fill_between(queries, lo, hi, alpha=0.2, color=color)
    ax.set_xlabel("queries asked")
    ax.set_ylabel("held-out accuracy (still-unlabeled points)")
    ax.set_title(
        f"RQ6: restart-disagreement vs. entropy (c={rq3.C}, {N_TRIALS} trials)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = "figures/rq6_restart_disagreement.png"
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
