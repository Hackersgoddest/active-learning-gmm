"""RQ5b: does TRUE class imbalance (not just the model's belief) make
uncertainty sampling lose to random sampling by MORE than RQ3's baseline
(43.3% win rate under balanced truth)?

    .venv/bin/python experiments/rq5b_true_imbalance.py

See research_log/RQ5b_true_imbalance.md for the design, and read it
alongside this file. This one reimplements RQ3's query loop (clamping,
warm-starting, the "uncertainty"/"random" branches) rather than extending
`rq3.run_query_loop` itself, because the metric has to change (balanced
accuracy, not plain accuracy; see the log's Section 5 for why plain accuracy is
actively misleading under imbalance).

Produces: a printed win-rate summary against the pre-committed hypothesis,
and figures/rq5b_true_imbalance.png.
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

N_MAJORITY = 540
N_MINORITY = 60  # 90/10 split, N=600 total, matching RQ1-RQ5's N
N_TRIALS = 30
N_RANDOM_SUBREPEATS = 5


def balanced_accuracy(pred, z_true, remaining):
    """Average accuracy on each class separately; fixes plain accuracy's
    trivial-majority trap under imbalance (research_log/RQ5b, Section 5)."""
    per_class = []
    for cls in (0, 1):
        mask = z_true[remaining] == cls
        if mask.sum() > 0:
            per_class.append((pred[remaining][mask] == cls).mean())
    return float(np.mean(per_class)) if per_class else 1.0


def run_query_loop_balanced(x, z_true, perm, pi0, mu0, Sigma0, strategy, rng):
    """Same mechanics as rq3.run_query_loop, metric swapped to balanced
    accuracy. See that function's docstring for what `perm` does."""
    n = len(x)
    clamped_labels = np.full(n, -1)
    pi, mu, Sigma = pi0.copy(), mu0.copy(), Sigma0.copy()
    inverse_perm = np.argsort(perm)
    accuracies = np.empty(rq3.QUERY_BUDGET)

    for step in range(rq3.QUERY_BUDGET):
        unlabeled = np.where(clamped_labels == -1)[0]

        if strategy == "uncertainty":
            r = responsibilities(x, pi, mu, Sigma)
            h = rq3.entropy(r)
            next_idx = unlabeled[np.argmax(h[unlabeled])]
        else:  # "random"
            next_idx = unlabeled[rng.integers(len(unlabeled))]

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
        pred = perm[np.argmax(r_final, axis=1)]
        accuracies[step] = balanced_accuracy(pred, z_true, remaining)

    return accuracies


def run_trial(trial_idx):
    data_seed = trial_idx + 30_000  # fresh seeds, different generator, not RQ3's data
    restart_seed = trial_idx + 40_000
    random_strategy_seed = trial_idx + 50_000

    x, z_true, mu_true = make_two_gaussians(
        rq3.C,
        sigma=rq3.SIGMA,
        n_per_component=(N_MAJORITY, N_MINORITY),
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

    uncertainty_curve = run_query_loop_balanced(
        x, z_true, perm, initial.pi, initial.mu, initial.Sigma, "uncertainty", None
    )

    rng_random = np.random.default_rng(random_strategy_seed)
    random_curves = np.array(
        [
            run_query_loop_balanced(
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
        "win": uncertainty_final
        > random_final_mean,  # "win" = uncertainty beats random
    }


def main():
    rows = [run_trial(t) for t in range(N_TRIALS)]

    n_wins = sum(r["win"] for r in rows)
    win_rate = n_wins / N_TRIALS
    print(
        f"Uncertainty sampling beat random sampling in {n_wins}/{N_TRIALS} "
        f"= {win_rate:.1%} of trials (90/10 true imbalance, balanced accuracy)"
    )
    print("For comparison, RQ3's win rate under balanced truth was 43.3%.")
    print("Predicted: uncertainty wins 30-40%.  'Wrong' line: more than 40%.")
    verdict = (
        "CONSISTENT with the hypothesis"
        if win_rate <= 0.40
        else "HYPOTHESIS WRONG (above the 40% line committed in advance)"
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
        0.5, color="gray", linestyle=":", label="chance level (balanced accuracy)"
    )
    ax.set_xlabel("queries asked")
    ax.set_ylabel("balanced accuracy (still-unlabeled points)")
    ax.set_title(
        f"RQ5b: uncertainty vs. random under 90/10 true imbalance "
        f"(c={rq3.C}, {N_TRIALS} trials)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = "figures/rq5b_true_imbalance.png"
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
