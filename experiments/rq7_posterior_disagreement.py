"""RQ7: does a real posterior over mu (via mean-field VI) provide a query
signal beyond plain entropy? This is the principled, Bayesian analogue of
RQ6's ad hoc restart-/bootstrap-disagreement ensembles.

    .venv/bin/python experiments/rq7_posterior_disagreement.py

The design is written up in research_log/RQ7_mcmc_and_vi.md, so read that
alongside this file. Reimplements RQ3's query loop for the same reason RQ6 did: the
new strategy needs extra per-step machinery (a fresh CAVI fit, sampling
from q(mu)) that doesn't fit cleanly into rq3.run_query_loop's existing
branches.

Produces: a printed win-rate summary against the pre-committed hypothesis,
and figures/rq7_posterior_disagreement.png.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import experiments.rq3_entropy_vs_random as rq3  # noqa: E402
from experiments.rq7_vi_meanfield import run_cavi  # noqa: E402
from gmmlab.datasets import make_two_gaussians  # noqa: E402
from gmmlab.em import fit, fit_best_of  # noqa: E402
from gmmlab.matching import match_components  # noqa: E402
from gmmlab.responsibilities import responsibilities  # noqa: E402

N_POSTERIOR_SAMPLES = 75
N_TRIALS = 30


def posterior_disagreement_scores(x, clamped_labels, pi, mu, Sigma, rng):
    """Fresh CAVI fit (pi, Sigma held fixed at this step's point estimate,
    clamped points held hard), then N_POSTERIOR_SAMPLES draws of mu from
    q(mu) = N(m_k, S_k). This is cheap since q(mu) is already a plain
    Gaussian, so there's no need to re-run CAVI per sample. Returns, per
    point, the variance of its responsibility across those samples: how
    much does this point's predicted cluster depend on which plausible mu
    you believe?"""
    m, S, _ = run_cavi(x, pi, mu, Sigma, clamped_labels=clamped_labels)

    n = len(x)
    r_samples = np.empty((N_POSTERIOR_SAMPLES, n))
    for s in range(N_POSTERIOR_SAMPLES):
        mu_sample = np.array(
            [rng.multivariate_normal(m[k], S[k]) for k in range(len(pi))]
        )
        r = responsibilities(x, pi, mu_sample, Sigma)
        r_samples[s] = r[:, 0]  # K=2: component 0's probability is enough

    return r_samples.var(axis=0)


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
        else:  # "posterior_disagreement"
            variance = posterior_disagreement_scores(
                x, clamped_labels, pi, mu, Sigma, rng
            )
            # ties are far less likely here than RQ6's discrete vote-count
            # scores (variance is continuous), but break them the same way
            # for consistency and to avoid any accidental array-order bias.
            v_unlabeled = variance[unlabeled]
            tied = unlabeled[v_unlabeled == v_unlabeled.max()]
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
    data_seed = trial_idx + 150_000
    restart_seed = trial_idx + 160_000
    posterior_seed = trial_idx + 170_000

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
        "posterior_disagreement",
        np.random.default_rng(posterior_seed),
    )

    entropy_final = entropy_curve[-1]
    disagreement_final = disagreement_curve[-1]

    return {
        "trial": trial_idx,
        "entropy_curve": entropy_curve,
        "disagreement_curve": disagreement_curve,
        "entropy_final": entropy_final,
        "disagreement_final": disagreement_final,
        "win": disagreement_final > entropy_final,
    }


def main():
    rows = [run_trial(t) for t in range(N_TRIALS)]

    n_wins = sum(r["win"] for r in rows)
    win_rate = n_wins / N_TRIALS
    print(
        f"Posterior-disagreement beat entropy in {n_wins}/{N_TRIALS} "
        f"= {win_rate:.1%} of trials"
    )
    print(
        "For comparison, RQ6's restart-disagreement won 56.7%, "
        "bootstrap-disagreement won 60.0%."
    )
    print("Predicted: 50-65%.  'Wrong' line: <=50% or >75%.")
    if win_rate <= 0.50:
        verdict = "HYPOTHESIS WRONG (at or below the 50% floor committed in advance)"
    elif win_rate > 0.75:
        verdict = "HYPOTHESIS WRONG (above the 75% ceiling committed in advance)"
    else:
        verdict = "CONSISTENT with the hypothesis"
    print(f"Verdict: {verdict}")

    entropy_all = np.array([r["entropy_curve"] for r in rows])
    disagreement_all = np.array([r["disagreement_curve"] for r in rows])
    queries = np.arange(1, rq3.QUERY_BUDGET + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, data, color in [
        ("entropy sampling", entropy_all, "C0"),
        ("posterior-disagreement sampling (VI)", disagreement_all, "C5"),
    ]:
        mean = data.mean(axis=0)
        lo = np.percentile(data, 25, axis=0)
        hi = np.percentile(data, 75, axis=0)
        ax.plot(queries, mean, label=label, color=color)
        ax.fill_between(queries, lo, hi, alpha=0.2, color=color)
    ax.set_xlabel("queries asked")
    ax.set_ylabel("held-out accuracy (still-unlabeled points)")
    ax.set_title(
        f"RQ7: posterior-disagreement vs. entropy (c={rq3.C}, {N_TRIALS} trials)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = "figures/rq7_posterior_disagreement.png"
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
