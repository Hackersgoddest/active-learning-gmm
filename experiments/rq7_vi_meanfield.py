"""RQ7, VI half: mean-field coordinate-ascent variational inference (CAVI)
for the same scoped Bayesian problem the Gibbs sampler solves. Posterior
over mu_1, mu_2 only; pi and Sigma stay fixed at EM's point estimates.

    .venv/bin/python experiments/rq7_vi_meanfield.py

Two alternating updates, no sampling anywhere (contrast with
rq7_gibbs_sampler.py, which samples both steps):
  1. soft responsibilities r_nk: EM's own E-step formula, plus one new
     term that discounts a point's confidence in cluster k by how
     uncertain we currently are about mu_k (-0.5*tr(Sigma_k^-1 @ S_k))
  2. q(mu_k) = N(m_k, S_k), the exact conjugate-Gaussian update the Gibbs
     sampler also uses, but with SOFT (fractional) counts r_nk instead of
     a hard 0/1 assignment

The ELBO is tracked every iteration and is guaranteed to never decrease
(same Jensen's-inequality guarantee as EM's log-likelihood, lessons/02),
checked the same way em.py's NonMonotonicLikelihoodError checks EM.

Validated two ways:
  (a) at c=4.0 (well-separated, no real ambiguity): q(mu_k)'s mean should
      land near EM's own mu_k, same sanity check rq7_gibbs_sampler.py used.
  (b) at c=2.5 (genuine ambiguity, ~200/600 points borderline): q(mu_k)
      should come out close to the Gibbs sampler's own empirical
      (mean, std) from rq7_z_uncertainty.py. Two independently-built
      methods (sampling vs. optimization) approximating the same posterior.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.rq7_gibbs_sampler import (  # noqa: E402
    BURN_IN,
    N_GIBBS_ITERS,
    PRIOR_MEAN,
    PRIOR_VAR,
    run_gibbs,
)
from gmmlab.datasets import make_two_gaussians  # noqa: E402
from gmmlab.em import _apply_clamping, fit_best_of  # noqa: E402

N_PER_COMPONENT = 300
SIGMA = 1.0
N_CAVI_ITERS = 100
ELBO_TOL = (
    1e-6  # floating-point slack, same spirit as em.py's NonMonotonicLikelihoodError
)


class NonMonotonicElboError(RuntimeError):
    pass


def soft_responsibilities(x, pi, m, S, Sigma, clamped_labels=None):
    """VI's step 1: like EM's E-step, with an added penalty for how
    uncertain we currently are about each mu_k (S[k]). `clamped_labels`
    (optional, same convention as gmmlab.em.fit: an (N,) int array, -1 for
    "not clamped") overwrites those rows with a hard one-hot vector, since
    a revealed true label isn't a belief to keep updating."""
    n, d = x.shape
    k_count = len(pi)
    log_r = np.empty((n, k_count))
    for k in range(k_count):
        sigma_k_inv = np.linalg.inv(Sigma[k])
        diff = x - m[k]
        quad = np.einsum("ni,ij,nj->n", diff, sigma_k_inv, diff)
        penalty = np.trace(sigma_k_inv @ S[k])
        log_r[:, k] = np.log(pi[k]) - 0.5 * quad - 0.5 * penalty
    log_r -= log_r.max(axis=1, keepdims=True)
    r = np.exp(log_r)
    r /= r.sum(axis=1, keepdims=True)
    if clamped_labels is not None:
        r = _apply_clamping(r, clamped_labels)
    return r


def mu_posterior_update(x, r, Sigma):
    """VI's step 2: the same conjugate-Gaussian formula
    rq7_gibbs_sampler.closed_form_posterior_given_fixed_z uses, but with
    soft (fractional) counts r[:, k] instead of a hard assignment."""
    k_count, d = Sigma.shape[0], x.shape[1]
    new_m = np.empty((k_count, d))
    new_S = np.empty((k_count, d, d))
    s0_inv = np.eye(d) / PRIOR_VAR
    for k in range(k_count):
        sigma_k_inv = np.linalg.inv(Sigma[k])
        n_k = r[:, k].sum()
        weighted_sum = (r[:, k : k + 1] * x).sum(axis=0)
        s_n_inv = s0_inv + n_k * sigma_k_inv
        s_n = np.linalg.inv(s_n_inv)
        m_n = s_n @ (s0_inv @ PRIOR_MEAN + sigma_k_inv @ weighted_sum)
        new_m[k] = m_n
        new_S[k] = s_n
    return new_m, new_S


def elbo(x, pi, m, S, Sigma, r):
    n, d = x.shape
    k_count = len(pi)
    s0 = PRIOR_VAR * np.eye(d)
    s0_inv = np.eye(d) / PRIOR_VAR
    _, logdet_s0 = np.linalg.slogdet(s0)
    total = 0.0
    for k in range(k_count):
        sigma_k_inv = np.linalg.inv(Sigma[k])
        _, logdet_sigma_k = np.linalg.slogdet(Sigma[k])
        diff = x - m[k]
        quad = np.einsum("ni,ij,nj->n", diff, sigma_k_inv, diff)
        penalty = np.trace(sigma_k_inv @ S[k])
        # E_q[log p(x_n, z_n=k | mu_k)] weighted by r_nk, summed over n
        total += np.sum(
            r[:, k]
            * (
                np.log(pi[k])
                - 0.5 * d * np.log(2 * np.pi)
                - 0.5 * logdet_sigma_k
                - 0.5 * (quad + penalty)
            )
        )
        # E_q[log p(mu_k)]
        diff_prior = m[k] - PRIOR_MEAN
        total += (
            -0.5 * d * np.log(2 * np.pi)
            - 0.5 * logdet_s0
            - 0.5 * (diff_prior @ s0_inv @ diff_prior + np.trace(s0_inv @ S[k]))
        )
        # entropy of q(mu_k)
        _, logdet_s_k = np.linalg.slogdet(S[k])
        total += 0.5 * d * np.log(2 * np.pi * np.e) + 0.5 * logdet_s_k
    # entropy of q(Z)
    r_safe = np.clip(r, 1e-300, 1.0)
    total += -np.sum(r * np.log(r_safe))
    return total


def run_cavi(
    x, pi, m_init, Sigma, n_iters=N_CAVI_ITERS, tol=ELBO_TOL, clamped_labels=None
):
    m, S = m_init.copy(), np.array([PRIOR_VAR * np.eye(x.shape[1]) for _ in pi])
    elbo_history = []
    for it in range(n_iters):
        r = soft_responsibilities(x, pi, m, S, Sigma, clamped_labels)
        m, S = mu_posterior_update(x, r, Sigma)
        current = elbo(x, pi, m, S, Sigma, r)
        # same convention as gmmlab.em.fit: once points are forced to a hard
        # label, the model is being asked to respect ground truth rather
        # than purely maximize the objective, so the monotonicity guarantee
        # no longer applies and isn't checked.
        if clamped_labels is None and elbo_history and current < elbo_history[-1] - tol:
            raise NonMonotonicElboError(
                f"ELBO decreased at iteration {it}: {elbo_history[-1]} -> {current}"
            )
        elbo_history.append(current)
    return m, S, elbo_history


def main():
    print("=== Validation (a): well-separated (c=4.0), vs. EM's point estimate ===")
    x, z_true, mu_true = make_two_gaussians(
        4.0, sigma=SIGMA, n_per_component=N_PER_COMPONENT, rng=np.random.default_rng(0)
    )
    initial = fit_best_of(
        x, 2, n_restarts=10, n_iter=2000, rng=np.random.default_rng(1)
    )
    m, S, elbo_history = run_cavi(x, initial.pi, initial.mu, initial.Sigma)
    print("EM mu:", initial.mu)
    print("VI posterior mean m:", m)
    print("VI posterior std (per dim):", np.sqrt(np.array([np.diag(Sk) for Sk in S])))
    print(f"max abs diff (VI mean vs EM mu): {np.abs(m - initial.mu).max():.4f}")
    print(f"ELBO monotonic increase confirmed over {len(elbo_history)} iterations")
    print(f"ELBO: first={elbo_history[0]:.3f}, last={elbo_history[-1]:.3f}")

    print("\n=== Validation (b): genuine ambiguity (c=2.5), vs. Gibbs sampler ===")
    x2, z_true2, mu_true2 = make_two_gaussians(
        2.5, sigma=SIGMA, n_per_component=N_PER_COMPONENT, rng=np.random.default_rng(0)
    )
    initial2 = fit_best_of(
        x2, 2, n_restarts=10, n_iter=2000, rng=np.random.default_rng(1)
    )
    m2, S2, elbo_history2 = run_cavi(x2, initial2.pi, initial2.mu, initial2.Sigma)
    print("VI posterior mean:", m2)
    print("VI posterior std:", np.sqrt(np.array([np.diag(Sk) for Sk in S2])))

    rng = np.random.default_rng(2)
    gibbs_samples = run_gibbs(
        x2, initial2.pi, initial2.mu, initial2.Sigma, rng, N_GIBBS_ITERS
    )
    gibbs_post = gibbs_samples[BURN_IN:]
    gibbs_mean = gibbs_post.mean(axis=0)
    gibbs_std = gibbs_post.std(axis=0)
    print("Gibbs posterior mean:", gibbs_mean)
    print("Gibbs posterior std:", gibbs_std)

    print(f"\nmax abs diff in means (VI vs Gibbs): {np.abs(m2 - gibbs_mean).max():.4f}")
    vi_std = np.sqrt(np.array([np.diag(Sk) for Sk in S2]))
    print(
        "std ratio (VI / Gibbs, expect close to but below 1 -- "
        f"VI is known to underestimate variance): \n{vi_std / gibbs_std}"
    )


if __name__ == "__main__":
    main()
