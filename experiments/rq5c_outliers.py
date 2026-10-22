"""RQ5c: does uncertainty sampling get fooled by low-density outliers that
look exactly as ambiguous as a real boundary point, but aren't informative?

    .venv/bin/python experiments/rq5c_outliers.py

The design lives in research_log/RQ5c_outliers.md, read that alongside
this file. This reimplements RQ3's query loop (clamping, warm-starting)
because the dataset itself now includes injected decoy points, and the
metric has to exclude them (held-out accuracy on the 600 real points only,
even though decoys remain queryable).

Produces: a printed win-rate summary against the pre-committed hypothesis,
a decoy-query-rate diagnostic, and figures/rq5c_outliers.png.
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

N_DECOYS = 5
DECOY_Y = 8.0  # ~8 std out, essentially impossible under the real generator
N_TRIALS = 30
N_RANDOM_SUBREPEATS = 5


def add_decoys(x, z_true, rng):
    """Append N_DECOYS points near x=0, y=DECOY_Y: same entropy as a real
    boundary point (x decides entropy here), and rare (y decides
    density). True label split arbitrarily (x=0 doesn't favor either real
    cluster anyway). Returns the augmented x, z_true, and a boolean mask
    (True = decoy) aligned with the augmented arrays.

    Spread independently (x: std=0.1, y: std=1.0 around DECOY_Y), not a
    tight jitter around one shared center. That came out of running this
    experiment and finding that identical coordinates (a true zero-variance
    cluster) AND even a small shared jitter (std=0.05, still a tight
    sub-cluster) both let EM collapse a near-singular covariance onto the
    decoys as a group. That's an even more extreme case of RQ1's
    unbounded-likelihood pathology than intended, and it twice pushed an
    iteration path through a configuration precise enough to trip
    NonMonotonicLikelihoodError (floating-point precision loss, not a logic
    bug; confirmed both times by reproducing it and checking it disappears
    without the decoys present). Spreading the 5 decoys apart from EACH
    OTHER means no single tight covariance can cover all of them: each one
    is still individually rare (y: comfortably >4 std in all but the most
    extreme tail) and still individually ambiguous (x stays close to 0,
    nowhere near the true means at +/-0.5), without the group being
    collapsible onto a single near-singular point. Stress-tested across all
    30 of RQ5c's trial seeds before trusting it (see conversation record).
    """
    decoy_x = np.column_stack(
        [rng.normal(scale=0.1, size=N_DECOYS), rng.normal(scale=1.0, size=N_DECOYS)]
    )
    decoy_x[:, 1] += DECOY_Y
    decoy_z = np.array([i % 2 for i in range(N_DECOYS)])

    x_aug = np.vstack([x, decoy_x])
    z_aug = np.concatenate([z_true, decoy_z])
    is_decoy = np.concatenate(
        [np.zeros(len(x), dtype=bool), np.ones(N_DECOYS, dtype=bool)]
    )
    return x_aug, z_aug, is_decoy


def run_query_loop_real_only(
    x, z_true, is_decoy, perm, pi0, mu0, Sigma0, strategy, rng
):
    """Same mechanics as rq3.run_query_loop: decoys are queryable (part of
    `unlabeled`) but excluded from the accuracy metric (real points only)."""
    n = len(x)
    clamped_labels = np.full(n, -1)
    pi, mu, Sigma = pi0.copy(), mu0.copy(), Sigma0.copy()
    inverse_perm = np.argsort(perm)
    accuracies = np.empty(rq3.QUERY_BUDGET)
    decoy_queries = 0

    for step in range(rq3.QUERY_BUDGET):
        unlabeled = np.where(clamped_labels == -1)[0]

        if strategy == "uncertainty":
            r = responsibilities(x, pi, mu, Sigma)
            h = rq3.entropy(r)
            next_idx = unlabeled[np.argmax(h[unlabeled])]
        else:  # "random"
            next_idx = unlabeled[rng.integers(len(unlabeled))]

        if is_decoy[next_idx]:
            decoy_queries += 1
        clamped_labels[next_idx] = int(inverse_perm[z_true[next_idx]])

        result = fit(
            x,
            2,
            n_iter=rq3.WARM_N_ITER,
            init_params=(pi, mu, Sigma),
            clamped_labels=clamped_labels,
        )
        pi, mu, Sigma = result.pi, result.mu, result.Sigma

        remaining_real = np.where((clamped_labels == -1) & ~is_decoy)[0]
        r_final = responsibilities(x, pi, mu, Sigma)
        pred = perm[np.argmax(r_final, axis=1)]
        accuracies[step] = (
            (pred[remaining_real] == z_true[remaining_real]).mean()
            if len(remaining_real)
            else 1.0
        )

    return accuracies, decoy_queries


def run_trial(trial_idx):
    data_seed = trial_idx + 60_000
    restart_seed = trial_idx + 70_000
    random_strategy_seed = trial_idx + 80_000

    x_real, z_real, mu_true = make_two_gaussians(
        rq3.C,
        sigma=rq3.SIGMA,
        n_per_component=rq3.N_PER_COMPONENT,
        rng=np.random.default_rng(data_seed),
    )
    x, z_true, is_decoy = add_decoys(
        x_real, z_real, np.random.default_rng(data_seed + 1)
    )

    initial = fit_best_of(
        x,
        2,
        n_restarts=rq3.N_RESTARTS,
        n_iter=rq3.INIT_N_ITER,
        rng=np.random.default_rng(restart_seed),
    )
    perm = match_components(initial.mu, mu_true)

    uncertainty_curve, uncertainty_decoys = run_query_loop_real_only(
        x,
        z_true,
        is_decoy,
        perm,
        initial.pi,
        initial.mu,
        initial.Sigma,
        "uncertainty",
        None,
    )

    rng_random = np.random.default_rng(random_strategy_seed)
    random_curves, random_decoys = [], []
    for _ in range(N_RANDOM_SUBREPEATS):
        curve, decoys = run_query_loop_real_only(
            x,
            z_true,
            is_decoy,
            perm,
            initial.pi,
            initial.mu,
            initial.Sigma,
            "random",
            rng_random,
        )
        random_curves.append(curve)
        random_decoys.append(decoys)
    random_curves = np.array(random_curves)

    uncertainty_final = uncertainty_curve[-1]
    random_final_mean = random_curves[:, -1].mean()

    return {
        "trial": trial_idx,
        "uncertainty_curve": uncertainty_curve,
        "random_curve_mean": random_curves.mean(axis=0),
        "uncertainty_final": uncertainty_final,
        "random_final_mean": random_final_mean,
        "win": uncertainty_final > random_final_mean,
        "uncertainty_decoy_queries": uncertainty_decoys,
        "random_decoy_queries_mean": float(np.mean(random_decoys)),
    }


def main():
    rows = [run_trial(t) for t in range(N_TRIALS)]

    n_wins = sum(r["win"] for r in rows)
    win_rate = n_wins / N_TRIALS
    print(
        f"Uncertainty sampling beat random sampling in {n_wins}/{N_TRIALS} "
        f"= {win_rate:.1%} of trials"
    )
    print("For comparison, RQ3's win rate (no decoys) was 43.3%.")
    print("Predicted: lower than 43.3%.  'Wrong' line: exceeds 40%.")
    verdict = (
        "CONSISTENT with the hypothesis"
        if win_rate <= 0.40
        else "HYPOTHESIS WRONG (above the 40% line committed in advance)"
    )
    print(f"Verdict: {verdict}")

    mean_unc_decoys = np.mean([r["uncertainty_decoy_queries"] for r in rows])
    mean_rand_decoys = np.mean([r["random_decoy_queries_mean"] for r in rows])
    chance_rate = N_DECOYS / (rq3.N_PER_COMPONENT * 2 + N_DECOYS) * rq3.QUERY_BUDGET
    print(
        f"\nMean decoy queries per trial (out of {rq3.QUERY_BUDGET}): "
        f"uncertainty={mean_unc_decoys:.2f}, random={mean_rand_decoys:.2f}, "
        f"pure-chance baseline={chance_rate:.2f}"
    )

    uncertainty_all = np.array([r["uncertainty_curve"] for r in rows])
    random_all = np.array([r["random_curve_mean"] for r in rows])
    queries = np.arange(1, rq3.QUERY_BUDGET + 1)

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
    ax.set_ylabel("accuracy on real points (still-unlabeled)")
    ax.set_title(
        f"RQ5c: uncertainty vs. random with {N_DECOYS} disguised outliers "
        f"(c={rq3.C}, {N_TRIALS} trials)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = "figures/rq5c_outliers.png"
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
