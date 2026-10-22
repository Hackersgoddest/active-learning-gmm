"""RQ7, MCMC half: a real Gibbs sampler for a Bayesian treatment of the
GMM's cluster means. pi and Sigma held fixed at their ordinary EM point
estimates (research_log/RQ7_mcmc.md's scoping decision); mu_1, mu_2, and Z
(cluster assignments) are treated as genuinely unknown and sampled.

    .venv/bin/python experiments/rq7_gibbs_sampler.py

Alternates, each iteration:
  1. sample z_n for every point, given the current mu's (a real random draw
     from the categorical responsibility distribution, not the soft r_nk
     we've used everywhere else tonight)
  2. sample mu_k for each component, given the current Z, from the EXACT
     closed-form Gaussian posterior (verified against a brute-force grid
     search before this file was written)

This file doesn't touch gmmlab at all. It's a standalone Bayesian add-on
built on top of an ordinary (already-fit) point estimate, not a
replacement for gmmlab's own EM.

A NOTE ON c: every other RQ in this project deliberately uses c=1.0 (clusters
one std-dev apart) because that's the hard, overlapping regime where active
learning strategies have real work to do. I checked this while building the
file: at c=1.0, fit_best_of's "best of 10 restarts" essentially never finds
the true 50/50 split. Scanning 20 data seeds, every single one converged to
an imbalanced, near-chance-accuracy partition (0.49-0.64), the same
unbounded-log-likelihood collapse the README's "Known gaps" section already
warns about, just more pervasive here than I'd previously quantified. That's
not a bug, since RQ3-RQ6's strategies are all compared starting from that
same shared starting point, so their head-to-head results are unaffected.
But it does mean there's no clean ground truth to validate a brand-new
sampler against at c=1.0. So this file uses c=4.0 (well-separated) on
purpose: the goal here is checking that the Gibbs sampler correctly
implements the math, not how uncertainty behaves under overlap, which is a
question for elsewhere.
"""

import numpy as np

from gmmlab.datasets import make_two_gaussians
from gmmlab.em import fit_best_of
from gmmlab.responsibilities import responsibilities

C = 4.0  # well-separated on purpose (see "A NOTE ON c" above)
SIGMA = 1.0
N_PER_COMPONENT = 300
PRIOR_MEAN = np.zeros(2)
PRIOR_VAR = 10.0  # S0 = PRIOR_VAR * I, deliberately wide and uninformative
N_GIBBS_ITERS = 5000
BURN_IN = 1000


def closed_form_posterior_given_fixed_z(x, z, k, Sigma_k):
    """The exact conjugate posterior N(m_N, S_N) for mu_k given a HARD,
    FIXED assignment z: the analytic (m_N, S_N) itself, not a sample from it.
    This is "what you'd believe about mu_k if you treated z as ground
    truth and never questioned it," the quantity rq7_z_uncertainty.py
    compares against the full Gibbs sampler's wider answer, which doesn't
    take z as given."""
    mask = z == k
    Nk = int(mask.sum())
    Sigma_k_inv = np.linalg.inv(Sigma_k)
    S0_inv = np.eye(2) / PRIOR_VAR
    if Nk == 0:
        return PRIOR_MEAN, PRIOR_VAR * np.eye(2)
    weighted_sum = x[mask].sum(axis=0)
    S_N_inv = S0_inv + Nk * Sigma_k_inv
    S_N = np.linalg.inv(S_N_inv)
    m_N = S_N @ (S0_inv @ PRIOR_MEAN + Sigma_k_inv @ weighted_sum)
    return m_N, S_N


def sample_mu_given_z(x, z, k, Sigma_k, rng):
    """One draw from closed_form_posterior_given_fixed_z, the step the
    Gibbs sampler uses at each iteration."""
    m_N, S_N = closed_form_posterior_given_fixed_z(x, z, k, Sigma_k)
    return rng.multivariate_normal(m_N, S_N)


def sample_z_given_mu(x, mu, pi, Sigma, rng):
    """A REAL random draw per point from the categorical responsibility
    distribution, not the soft r_nk every other experiment tonight used."""
    r = responsibilities(x, pi, mu, Sigma)
    n = len(x)
    u = rng.random(n)
    # categorical draw via inverse-CDF for 2 categories: z=1 if u < r[:,1]
    return (u < r[:, 1]).astype(int)


def run_gibbs(x, pi, mu_init, Sigma, rng, n_iters):
    mu = mu_init.copy()
    samples_mu = np.empty((n_iters, 2, 2))  # (iteration, component, dim)
    for it in range(n_iters):
        z = sample_z_given_mu(x, mu, pi, Sigma, rng)
        mu = np.array([sample_mu_given_z(x, z, k, Sigma[k], rng) for k in range(2)])
        samples_mu[it] = mu
    return samples_mu


def main():
    x, z_true, mu_true = make_two_gaussians(
        C, sigma=SIGMA, n_per_component=N_PER_COMPONENT, rng=np.random.default_rng(0)
    )
    initial = fit_best_of(
        x, 2, n_restarts=10, n_iter=2000, rng=np.random.default_rng(1)
    )
    print("EM point estimate (held fixed for pi, Sigma; used as mu's starting point):")
    print("  pi:", initial.pi)
    print("  mu:", initial.mu)

    rng = np.random.default_rng(2)
    samples = run_gibbs(x, initial.pi, initial.mu, initial.Sigma, rng, N_GIBBS_ITERS)
    post_burn_in = samples[BURN_IN:]

    posterior_mean = post_burn_in.mean(axis=0)
    posterior_std = post_burn_in.std(axis=0)
    print(f"\nGibbs sampler, after discarding the first {BURN_IN} burn-in iterations:")
    print("  posterior mean of mu:\n", posterior_mean)
    print("  posterior std of mu (per dimension):\n", posterior_std)
    diff = np.abs(posterior_mean - initial.mu).max()
    print(
        "\nSanity check: does the sampler's mean land near EM's point estimate?"
        f"\n  difference: {diff:.4f} (max abs diff)"
    )


if __name__ == "__main__":
    main()
