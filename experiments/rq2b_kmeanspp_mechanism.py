"""RQ2 follow-up: when k-means++ init produces a lopsided fit, did its SECOND
chosen point land inside the eventual small component more often than a
single random grab from the data would?

    .venv/bin/python experiments/rq2b_kmeanspp_mechanism.py

Hypothesis (research_log/RQ2_initialization.md, Section 11, sharpened before this
ran): if the first (uniformly random) point lands in the big group, k-means++
distance-weighting should pull the second point toward the small group;
if the first point already landed in the small group, the weighting should
pull the second point toward the big group instead. Prediction: ~80% of
lopsided trials, not just >baseline. Wrong if the real rate is below 60%.

Reuses RQ2's exact per-trial seeds (data_seed = trial_idx, restart_seed =
trial_idx + 10_000) and lopsided threshold, so "which trials are lopsided"
matches research_log/RQ2_initialization.md's own table exactly; this file
does not re-decide that, it only asks a new question about the SAME trials.
"""

import numpy as np

from gmmlab.datasets import make_two_gaussians
from gmmlab.em import best_of, fit_many
from gmmlab.responsibilities import responsibilities

C_TRIALS = [(0.5, 30), (1.0, 30)]  # same as RQ2, minus the c=4 control
# (c=4 has ~0/10 lopsided trials in RQ2, so there's nothing to measure a mechanism on)
SIGMA = 1.0
N_PER_COMPONENT = 300
N_RESTARTS = 10
N_ITER = 2000
LOPSIDED_THRESHOLD = 0.10  # identical to RQ2, same definition of "lopsided"


def run_trial(c, trial_idx):
    data_seed = trial_idx
    restart_seed = trial_idx + 10_000
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
        init="kmeans++",
    )
    winner = best_of(results)

    if winner.pi.min() >= LOPSIDED_THRESHOLD:
        return None  # not a lopsided trial, nothing to check

    small_component = int(np.argmin(winner.pi))
    r = responsibilities(x, winner.pi, winner.mu, winner.Sigma)
    hard_assignment = np.argmax(r, axis=1)

    second_pick_idx = int(winner.init_indices[1])
    second_landed_in_small = bool(hard_assignment[second_pick_idx] == small_component)
    baseline_rate = float((hard_assignment == small_component).mean())

    return {
        "c": c,
        "trial": trial_idx,
        "min_weight": float(winner.pi.min()),
        "second_landed_in_small": second_landed_in_small,
        "baseline_rate": baseline_rate,
    }


def main():
    records = []
    for c, n_trials in C_TRIALS:
        for t in range(n_trials):
            rec = run_trial(c, t)
            if rec is not None:
                records.append(rec)

    n_lopsided = len(records)
    n_landed = sum(r["second_landed_in_small"] for r in records)
    observed_rate = n_landed / n_lopsided
    mean_baseline = float(np.mean([r["baseline_rate"] for r in records]))

    print(f"Lopsided kmeans++ trials found: {n_lopsided}")
    print(
        f"k-means++'s 2nd point landed in the small component: "
        f"{n_landed}/{n_lopsided} = {observed_rate:.1%}"
    )
    print(f"Mean per-trial baseline rate (pure chance): {mean_baseline:.1%}")
    print()
    print("Predicted: ~80%.  'Wrong' line: below 60%.")
    verdict = (
        "CONSISTENT with the mechanism"
        if observed_rate >= 0.60
        else "HYPOTHESIS WRONG (below the 60% line committed in advance)"
    )
    print(f"Verdict: {verdict}")

    for c, _ in C_TRIALS:
        c_records = [r for r in records if r["c"] == c]
        if not c_records:
            continue
        c_landed = sum(r["second_landed_in_small"] for r in c_records)
        print(
            f"  c={c}: {c_landed}/{len(c_records)} = "
            f"{c_landed / len(c_records):.1%} "
            f"(n={len(c_records)} lopsided trials)"
        )


if __name__ == "__main__":
    main()
