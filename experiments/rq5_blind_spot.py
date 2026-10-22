"""RQ5: is the confidently-wrong blind spot (Lesson 4, Section 4.4) a real,
additional cost beyond the general redundancy problem RQ3/RQ4 already
found?

    .venv/bin/python experiments/rq5_blind_spot.py

The design for this one is in research_log/RQ5_blind_spot.md; read that
alongside this file. It reuses experiments/rq3_entropy_vs_random.py's
`run_query_loop` for both conditions (the "oracle_wrong" strategy was added
there, additively, and verified not to change the existing
"uncertainty"/"random" behaviour).

Produces: a printed win-rate summary against the pre-committed hypothesis,
and figures/rq5_blind_spot.png.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import experiments.rq3_entropy_vs_random as rq3  # noqa: E402
from gmmlab.datasets import make_two_gaussians  # noqa: E402
from gmmlab.em import fit_best_of  # noqa: E402
from gmmlab.matching import match_components  # noqa: E402

LOPSIDED_THRESHOLD = 0.10  # same rule as RQ2/RQ3
N_TRIALS = 30


def find_lopsided_trials(n_needed):
    """Scan candidate trial indices (RQ3's exact seed scheme) until n_needed
    severely-lopsided initial fits are found. Not a different population,
    just a filtered subset of the same one RQ3 used."""
    found = []
    candidate_idx = 0
    while len(found) < n_needed:
        data_seed = candidate_idx
        restart_seed = candidate_idx + 10_000
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
        if initial.pi.min() < LOPSIDED_THRESHOLD:
            perm = match_components(initial.mu, mu_true)
            found.append((candidate_idx, x, z_true, perm, initial))
        candidate_idx += 1
    print(f"Scanned {candidate_idx} candidates to find {n_needed} lopsided trials.")
    return found


def run_trial(x, z_true, perm, initial):
    uncertainty_curve = rq3.run_query_loop(
        x, z_true, perm, initial.pi, initial.mu, initial.Sigma, "uncertainty", None
    )
    oracle_curve = rq3.run_query_loop(
        x, z_true, perm, initial.pi, initial.mu, initial.Sigma, "oracle_wrong", None
    )
    uncertainty_final = uncertainty_curve[-1]
    oracle_final = oracle_curve[-1]
    return {
        "uncertainty_curve": uncertainty_curve,
        "oracle_curve": oracle_curve,
        "uncertainty_final": uncertainty_final,
        "oracle_final": oracle_final,
        "win": oracle_final > uncertainty_final,  # "win" = oracle beats uncertainty
    }


def main():
    trials = find_lopsided_trials(N_TRIALS)
    rows = [
        run_trial(x, z_true, perm, initial) for _, x, z_true, perm, initial in trials
    ]

    n_wins = sum(r["win"] for r in rows)
    win_rate = n_wins / N_TRIALS
    print(
        f"Oracle beat real uncertainty sampling in {n_wins}/{N_TRIALS} "
        f"= {win_rate:.1%} of severely-lopsided-start trials"
    )
    print("Predicted: 60-70%.  'Wrong' line: below 50%.")
    verdict = (
        "CONSISTENT with the hypothesis"
        if win_rate >= 0.50
        else "HYPOTHESIS WRONG (below the 50% line committed in advance)"
    )
    print(f"Verdict: {verdict}")

    uncertainty_all = np.array([r["uncertainty_curve"] for r in rows])
    oracle_all = np.array([r["oracle_curve"] for r in rows])
    queries = np.arange(1, rq3.QUERY_BUDGET + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, data, color in [
        ("real uncertainty sampling", uncertainty_all, "C0"),
        ("confidently-wrong oracle", oracle_all, "C3"),
    ]:
        mean = data.mean(axis=0)
        lo = np.percentile(data, 25, axis=0)
        hi = np.percentile(data, 75, axis=0)
        ax.plot(queries, mean, label=label, color=color)
        ax.fill_between(queries, lo, hi, alpha=0.2, color=color)
    ax.set_xlabel("queries asked")
    ax.set_ylabel("held-out accuracy (still-unlabeled points)")
    ax.set_title(
        f"RQ5: is the confidently-wrong blind spot real? "
        f"(c={rq3.C}, {N_TRIALS} severely-lopsided-start trials)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = "figures/rq5_blind_spot.png"
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
