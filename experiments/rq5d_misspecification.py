"""RQ5d: what does model misspecification (K=2 fit to 3 true clusters)
cost, and does it change the uncertainty-vs-random comparison?

    .venv/bin/python experiments/rq5d_misspecification.py

See research_log/RQ5d_misspecification.md for the design and read it
alongside this file. The query loop here is reimplemented (clamping,
warm-starting) rather than reused, because both the data (3 true clusters,
not 2) and the metric (majority-vote per fitted component against all 3
true labels, not a 2-component perm match) differ fundamentally from RQ3's.

Produces: a printed win-rate summary and ceiling check against the
pre-committed hypothesis, and figures/rq5d_misspecification.png.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import experiments.rq3_entropy_vs_random as rq3  # noqa: E402
from gmmlab.datasets import make_two_gaussians  # noqa: E402
from gmmlab.em import fit, fit_best_of  # noqa: E402
from gmmlab.responsibilities import responsibilities  # noqa: E402

N_PER_CLUSTER = 300  # Left, Middle, Right; 900 total
LEFT_X = -5.0  # ~4.5 separation from the nearer of Middle/Right (+/-0.5)
PREDICTED_CEILING = (N_PER_CLUSTER + N_PER_CLUSTER) / (3 * N_PER_CLUSTER)  # 2/3
N_TRIALS = 30
N_RANDOM_SUBREPEATS = 5


def make_three_clusters(rng):
    """0 = Left, 1 = Middle, 2 = Right. Middle/Right use RQ1-RQ5's own
    c=1.0 two-cluster generator; Left is a third, well-separated cluster
    added on top."""
    middle_right_x, middle_right_z, _ = make_two_gaussians(
        rq3.C, sigma=rq3.SIGMA, n_per_component=N_PER_CLUSTER, rng=rng
    )
    left_x = np.column_stack(
        [
            LEFT_X + rng.normal(scale=rq3.SIGMA, size=N_PER_CLUSTER),
            rng.normal(scale=rq3.SIGMA, size=N_PER_CLUSTER),
        ]
    )
    x = np.vstack([left_x, middle_right_x])
    # middle_right_z is 0/1 for the two overlapping clusters -> relabel as
    # true clusters 1 (Middle) and 2 (Right); Left is true cluster 0
    z_true3 = np.concatenate([np.zeros(N_PER_CLUSTER, dtype=int), middle_right_z + 1])
    return x, z_true3


def majority_identity(pred, z_true3, k_count=2):
    """Which true label (0/1/2) is most common among the points each
    fitted component currently claims (hard assignment, argmax)."""
    identity = np.full(k_count, -1)
    for k in range(k_count):
        mask = pred == k
        if mask.sum() > 0:
            identity[k] = np.argmax(np.bincount(z_true3[mask], minlength=3))
    return identity


def true_label_to_component_map(identity):
    """Left (true label 0) goes to whichever component's majority identity
    is 0; Middle and Right (1, 2) BOTH go to the other component. There's
    no dedicated place for that distinction, by construction."""
    left_component = int(np.where(identity == 0)[0][0])
    other_component = 1 - left_component
    return {0: left_component, 1: other_component, 2: other_component}


def multiway_accuracy(pred, z_true3, identity, remaining):
    if len(remaining) == 0:
        return 1.0
    correct = identity[pred[remaining]] == z_true3[remaining]
    return float(correct.mean())


def run_query_loop(x, z_true3, pi0, mu0, Sigma0, strategy, rng):
    n = len(x)
    clamped_labels = np.full(n, -1)
    pi, mu, Sigma = pi0.copy(), mu0.copy(), Sigma0.copy()
    accuracies = np.empty(rq3.QUERY_BUDGET)

    # fixed once, right after the initial (unclamped) fit, same stability
    # principle as `perm` in RQ3-RQ5c
    r0 = responsibilities(x, pi, mu, Sigma)
    pred0 = np.argmax(r0, axis=1)
    identity = majority_identity(pred0, z_true3)
    label_to_component = true_label_to_component_map(identity)

    for step in range(rq3.QUERY_BUDGET):
        unlabeled = np.where(clamped_labels == -1)[0]

        if strategy == "uncertainty":
            r = responsibilities(x, pi, mu, Sigma)
            h = rq3.entropy(r)
            next_idx = unlabeled[np.argmax(h[unlabeled])]
        else:  # "random"
            next_idx = unlabeled[rng.integers(len(unlabeled))]

        clamped_labels[next_idx] = label_to_component[int(z_true3[next_idx])]

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
        pred_final = np.argmax(r_final, axis=1)
        accuracies[step] = multiway_accuracy(pred_final, z_true3, identity, remaining)

    return accuracies


def run_trial(trial_idx):
    data_seed = trial_idx + 90_000
    restart_seed = trial_idx + 100_000
    random_strategy_seed = trial_idx + 110_000

    x, z_true3 = make_three_clusters(np.random.default_rng(data_seed))
    initial = fit_best_of(
        x,
        2,
        n_restarts=rq3.N_RESTARTS,
        n_iter=rq3.INIT_N_ITER,
        rng=np.random.default_rng(restart_seed),
    )

    uncertainty_curve = run_query_loop(
        x, z_true3, initial.pi, initial.mu, initial.Sigma, "uncertainty", None
    )

    rng_random = np.random.default_rng(random_strategy_seed)
    random_curves = np.array(
        [
            run_query_loop(
                x, z_true3, initial.pi, initial.mu, initial.Sigma, "random", rng_random
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
    }


def main():
    rows = [run_trial(t) for t in range(N_TRIALS)]

    n_wins = sum(r["win"] for r in rows)
    win_rate = n_wins / N_TRIALS
    mean_final_acc = np.mean(
        [r["uncertainty_final"] for r in rows] + [r["random_final_mean"] for r in rows]
    )

    print(f"Predicted ceiling: {PREDICTED_CEILING:.1%}")
    print(
        f"Mean achieved final accuracy (both strategies pooled): {mean_final_acc:.1%}"
    )
    print()
    print(
        f"Uncertainty sampling beat random sampling in {n_wins}/{N_TRIALS} "
        f"= {win_rate:.1%} of trials"
    )
    print("For comparison, RQ3's win rate (no misspecification) was 43.3%.")
    print("Predicted: 40-45%.  'Wrong' line: below 40% or above 45%.")
    verdict = (
        "CONSISTENT with the hypothesis"
        if 0.40 <= win_rate <= 0.45
        else "HYPOTHESIS WRONG (outside the 40-45% range committed in advance)"
    )
    print(f"Verdict: {verdict}")

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
    ax.axhline(
        PREDICTED_CEILING,
        color="gray",
        linestyle=":",
        label=f"predicted ceiling ({PREDICTED_CEILING:.1%})",
    )
    ax.set_xlabel("queries asked")
    ax.set_ylabel("multiway accuracy (still-unlabeled points)")
    ax.set_title(
        f"RQ5d: misspecification (K=2 fit to 3 true clusters), {N_TRIALS} trials"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = "figures/rq5d_misspecification.png"
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
