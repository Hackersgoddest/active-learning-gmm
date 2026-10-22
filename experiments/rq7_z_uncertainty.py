"""RQ7 follow-up: does the full Gibbs posterior (which also treats Z as
unknown) come out WIDER than the naive fixed-Z closed form (which trusts
EM's hard assignment as ground truth and never questions it)?

    .venv/bin/python experiments/rq7_z_uncertainty.py

rq7_gibbs_sampler.py validated the sampler at c=4.0 (well-separated), where
Z is essentially certain for every point, so the comparison there was
trivial: the Gibbs posterior and the fixed-Z posterior had to agree,
because there was nothing for Z-uncertainty to add. This file uses c=2.5
instead, where EM still recovers a non-collapsed 2-cluster structure
(checked it directly: pi=[0.505,0.495], 89.5% accuracy against true
labels), but ~200/600 points have soft responsibility between 0.1 and 0.9.
That's real, unresolved ambiguity about which cluster they belong to, and
the regime where the two posteriors should actually differ.
"""

import numpy as np
from rq7_gibbs_sampler import (
    BURN_IN,
    N_GIBBS_ITERS,
    closed_form_posterior_given_fixed_z,
    run_gibbs,
)

from gmmlab.datasets import make_two_gaussians
from gmmlab.em import fit_best_of

C = 2.5
SIGMA = 1.0
N_PER_COMPONENT = 300


def main():
    x, z_true, mu_true = make_two_gaussians(
        C, sigma=SIGMA, n_per_component=N_PER_COMPONENT, rng=np.random.default_rng(0)
    )
    initial = fit_best_of(
        x, 2, n_restarts=10, n_iter=2000, rng=np.random.default_rng(1)
    )
    print("EM point estimate:")
    print("  pi:", initial.pi)
    print("  mu:", initial.mu)

    # The naive fixed-Z posterior: harden EM's own final soft responsibilities
    # into one hard Z (argmax), then trust it completely. No re-sampling, no
    # acknowledging that z could have gone the other way for borderline
    # points. This is "what you'd believe if you stopped at EM's own answer."
    from gmmlab.responsibilities import responsibilities

    r = responsibilities(x, initial.pi, initial.mu, initial.Sigma)
    z_hardened = np.argmax(r, axis=1)
    n_ambiguous = int(((r[:, 0] > 0.1) & (r[:, 0] < 0.9)).sum())
    print(
        f"\n{n_ambiguous}/600 points have genuinely ambiguous responsibility (0.1-0.9)."
    )

    print("\nNaive fixed-Z posterior (trusts the hardened Z completely):")
    fixed_z_std = np.empty((2, 2))
    for k in range(2):
        m_N, S_N = closed_form_posterior_given_fixed_z(
            x, z_hardened, k, initial.Sigma[k]
        )
        fixed_z_std[k] = np.sqrt(np.diag(S_N))
        print(f"  component {k}: mean={m_N}, std={fixed_z_std[k]}")

    # The full Gibbs sampler: resamples Z fresh every iteration, so its
    # answer also reflects genuine uncertainty about which cluster each
    # ambiguous point belongs to.
    rng = np.random.default_rng(2)
    samples = run_gibbs(x, initial.pi, initial.mu, initial.Sigma, rng, N_GIBBS_ITERS)
    post = samples[BURN_IN:]
    gibbs_mean = post.mean(axis=0)
    gibbs_std = post.std(axis=0)

    print("\nFull Gibbs posterior (Z resampled every iteration):")
    for k in range(2):
        print(f"  component {k}: mean={gibbs_mean[k]}, std={gibbs_std[k]}")

    print(
        "\nComparison (std ratio, Gibbs / fixed-Z -- "
        "expect > 1 if Z-uncertainty matters):"
    )
    ratio = gibbs_std / fixed_z_std
    for k in range(2):
        print(f"  component {k}: {ratio[k]}")


if __name__ == "__main__":
    main()
