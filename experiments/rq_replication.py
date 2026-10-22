"""Replication with more trials, on fresh seeds, for every head-to-head result
that was inconclusive at 30 trials, plus RQ7, the strongest result.

    .venv/bin/python experiments/rq_replication.py

Why this exists: an exact binomial test on the original 30-trial runs showed
that five of the nine comparisons (RQ3, RQ4, RQ5a, RQ6, RQ6c) could not be
told apart from a coin flip. Thirty trials only has the power to detect large
effects. This script reruns each comparison unchanged, through each
experiment's own run_trial, on 100 new trials whose seeds don't overlap the
originals (trial indices 30-129; RQ5a scans for lopsided starts beginning at
candidate 1000).

Replication criterion, written down before running: an effect counts as
replicated if the new win rate lands on the same side of 50% as the original
AND a two-sided sign test on the new trials (ties dropped) gives p < 0.05.
If the sign test misses, the result stays inconclusive, whatever the win
rate looks like. The original 30-trial verdicts are left as they were
pre-registered; this is an addendum, not a re-scoring.

Writes data/replication_results.json and prints a summary table.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

N_NEW = 100
FIRST_NEW_TRIAL = 30
RQ5A_FIRST_CANDIDATE = 1000
OUT = Path(__file__).resolve().parent.parent / "data" / "replication_results.json"

# experiment -> (module, key for strategy A, key for strategy B); a win means A > B
EXPERIMENTS = {
    "RQ3": ("rq3_entropy_vs_random", "uncertainty_final", "random_final_mean"),
    "RQ4": ("rq4_batch_entropy", "sequential_final", "batch_final"),
    "RQ5a": ("rq5_blind_spot", "oracle_final", "uncertainty_final"),
    "RQ6": ("rq6_restart_disagreement", "disagreement_final", "entropy_final"),
    "RQ6c": ("rq6c_bootstrap_disagreement", "disagreement_final", "entropy_final"),
    "RQ7": ("rq7_posterior_disagreement", "disagreement_final", "entropy_final"),
}
ORIGINAL_WINS = {"RQ3": 13, "RQ4": 20, "RQ5a": 12, "RQ6": 17, "RQ6c": 18, "RQ7": 24}


def _module(name):
    import importlib

    return importlib.import_module(f"experiments.{name}")


def rq5a_initial(candidate_idx):
    """Rebuild one RQ5a candidate exactly the way find_lopsided_trials does."""
    rq5 = _module("rq5_blind_spot")
    from gmmlab.datasets import make_two_gaussians
    from gmmlab.em import fit_best_of
    from gmmlab.matching import match_components

    rq3 = rq5.rq3
    x, z_true, mu_true = make_two_gaussians(
        rq3.C,
        sigma=rq3.SIGMA,
        n_per_component=rq3.N_PER_COMPONENT,
        rng=np.random.default_rng(candidate_idx),
    )
    initial = fit_best_of(
        x,
        2,
        n_restarts=rq3.N_RESTARTS,
        n_iter=rq3.INIT_N_ITER,
        rng=np.random.default_rng(candidate_idx + 10_000),
    )
    lopsided = initial.pi.min() < rq5.LOPSIDED_THRESHOLD
    perm = match_components(initial.mu, mu_true)
    return lopsided, x, z_true, perm, initial


def is_rq5a_candidate(candidate_idx):
    return candidate_idx, bool(rq5a_initial(candidate_idx)[0])


def run_one(exp, trial_idx):
    module_name, key_a, key_b = EXPERIMENTS[exp]
    mod = _module(module_name)
    if exp == "RQ5a":
        _, x, z_true, perm, initial = rq5a_initial(trial_idx)
        row = mod.run_trial(x, z_true, perm, initial)
    else:
        row = mod.run_trial(trial_idx)
    return exp, trial_idx, float(row[key_a]), float(row[key_b])


def summarize(a, b):
    a, b = np.array(a), np.array(b)
    wins, losses = int((a > b).sum()), int((a < b).sum())
    ties = len(a) - wins - losses
    sign = binomtest(wins, wins + losses, 0.5) if wins + losses else None
    ci = binomtest(wins, len(a), 0.5).proportion_ci(0.95)
    return {
        "n": len(a),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": wins / len(a),
        "win_rate_ci95": [ci.low, ci.high],
        "sign_test_p": sign.pvalue if sign else None,
        "mean_accuracy_gap": float((a - b).mean()),
    }


def main():
    workers = max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rq5a_trials, candidate = [], RQ5A_FIRST_CANDIDATE
        while len(rq5a_trials) < N_NEW:
            batch = range(candidate, candidate + 200)
            flags = dict(pool.map(is_rq5a_candidate, batch))
            rq5a_trials += [c for c in batch if flags[c]]
            candidate += 200
        rq5a_trials = rq5a_trials[:N_NEW]
        last = rq5a_trials[-1]
        print(f"RQ5a: first {N_NEW} lopsided candidates found by index {last}")

        jobs = []
        for exp in EXPERIMENTS:
            trials = (
                rq5a_trials
                if exp == "RQ5a"
                else range(FIRST_NEW_TRIAL, FIRST_NEW_TRIAL + N_NEW)
            )
            jobs += [(exp, t) for t in trials]

        results = {exp: [] for exp in EXPERIMENTS}
        futures = [pool.submit(run_one, exp, t) for exp, t in jobs]
        for done, fut in enumerate(as_completed(futures), 1):
            exp, t, a, b = fut.result()
            results[exp].append({"trial": t, "a": a, "b": b})
            if done % 25 == 0 or done == len(futures):
                print(f"{done}/{len(futures)} trials finished", flush=True)

    summary = {}
    print(
        f"\n{'RQ':5s} {'orig (n=30)':>12s} {'new wins':>9s} {'ties':>5s} "
        f"{'win rate [95% CI]':>22s} {'sign p':>8s}  verdict"
    )
    for exp, rows in results.items():
        rows.sort(key=lambda r: r["trial"])
        s = summarize([r["a"] for r in rows], [r["b"] for r in rows])
        same_side = (s["win_rate"] > 0.5) == (ORIGINAL_WINS[exp] / 30 > 0.5)
        p = s["sign_test_p"]
        s["replicated"] = bool(same_side and p is not None and p < 0.05)
        summary[exp] = {"original_wins_of_30": ORIGINAL_WINS[exp], **s, "trials": rows}
        verdict = "replicated" if s["replicated"] else "not replicated / inconclusive"
        lo, hi = s["win_rate_ci95"]
        print(
            f"{exp:5s} {ORIGINAL_WINS[exp]:>9d}/30 {s['wins']:>6d}/{s['n']} "
            f"{s['ties']:>5d} {s['win_rate']:>8.1%} [{lo:.2f}, {hi:.2f}] "
            f"{p:>8.4f}  {verdict}"
        )

    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
