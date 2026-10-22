# RQ7: What changes when we place a posterior distribution over model parameters?

- **Date started / run:** 2026-10-21
- **Status:** done. Both halves ("do one for each", a real MCMC sampler and
  a real VI implementation) built and cross-validated; the resulting active
  learning hypothesis was wrong, on the high side.
- **Replication (added 2026-10-21):** Replicated at a smaller size at 100 new trials (59/100, p = 0.04); see Section 12.
- **Code:** `experiments/rq7_gibbs_sampler.py` (MCMC), `experiments/rq7_vi_meanfield.py`
  (VI), `experiments/rq7_z_uncertainty.py` (the two compared against each
  other and against a naive baseline), `experiments/rq7_posterior_disagreement.py`
  (the active-learning experiment). **Figure:** `figures/rq7_posterior_disagreement.png`

## Plain-English summary (read this first; Sections 1–6 below are the technical record)

Every RQ so far treated the model's parameters (π, μ, Σ) as single best-guess
numbers (point estimates from EM) and asked how *responsibility* entropy,
meaning uncertainty about which cluster a point belongs to given those fixed
numbers, behaves as a query signal. RQ7 asks a different question: what if the
parameters themselves are uncertain too? Scoped deliberately (agreed in
conversation before any code): hold π and Σ fixed at EM's point estimates,
and treat only the two cluster means, μ₁ and μ₂, as genuinely unknown,
with a Gaussian prior and an exact conjugate posterior available as ground
truth when Z (cluster assignment) is also held fixed. This scoping keeps an
exact closed form available to validate against, at the cost of not being a
full Bayesian treatment of the whole model.

Two independent ways of approximating the true posterior over (μ₁, μ₂) when
Z is *not* held fixed (a genuinely harder problem than the fixed-Z closed
form, since now there's real uncertainty about which points belong to which
cluster) were built and checked against each other:

- **MCMC (Gibbs sampling):** alternates sampling each point's hard cluster
  assignment, then sampling μ₁, μ₂ from the exact conjugate posterior given
  that assignment, thousands of times.
- **VI (mean-field CAVI):** no sampling at all, it alternates computing the
  single best-fitting soft-responsibility distribution and the single
  best-fitting Gaussian belief about μ₁, μ₂, holding each fixed while
  optimizing the other, maximizing the ELBO (Lesson 2's own quantity, same
  Jensen's-inequality guarantee, now applied to μ instead of Z).

Both were validated at a well-separated setting (c=4.0, where Z is
essentially certain) against EM's own point estimate, and against each
other at a setting with genuine ambiguity (c=2.5, ~200/600 points
borderline). They agreed closely on the posterior mean (within 0.015) and
disagreed on the width in exactly the theoretically predicted direction: VI
systematically underestimated the posterior's spread (80-90% of Gibbs's
std), the textbook consequence of forcing an independent (mean-field)
approximation onto variables (Z and μ) whose true posterior is entangled.

One unplanned but important discovery while picking a validation seed:
**at the project's usual c=1.0, EM's "best of 10 restarts" essentially
never recovers the true balanced clusters.** Scanned 20 data seeds, and
every single one converged to an imbalanced, near-chance-accuracy initial fit
(0.49-0.64). This is the known unbounded-log-likelihood collapse (README's
"Known gaps"), just far more pervasive at this separation than previously
quantified. Not a bug in RQ3-RQ6, since both compared strategies always start
from that same shared point), but the reason this file's validation work
uses larger c values instead.

Built on top of the validated VI machinery: a new active-learning strategy,
**posterior-disagreement**, the principled, Bayesian analogue of RQ6's ad
hoc restart-/bootstrap-disagreement ensembles. At every query step, run VI
fresh, draw samples directly from its Gaussian q(μ), and query whichever
point's predicted cluster varies most across those samples. In other words,
query whichever point's classification depends most on which plausible μ
you believe, as opposed to entropy's "ambiguous under one fixed μ."
Pre-registered prediction: similar magnitude to RQ6's ensembles (50-65%),
since it's the same underlying mechanism with a principled posterior
standing in for an ad hoc ensemble. **Actual result: 80.0% (24/30), above
the 75% ceiling. The hypothesis was wrong, by a wide margin, in the winning
direction.**

## 1. Research question

(Locked wording, README.) What changes when we place a posterior
distribution over model parameters?

## 2. Hypothesis

**Validation hypotheses (implicit, numerical, committed before each check
ran):** the Gibbs sampler's posterior mean should land near EM's point
estimate at a well-separated setting (no formal threshold set, just "close,"
in the same sense RQ1-RQ6's own sanity checks used); the VI posterior mean
should agree closely with the Gibbs posterior mean, and VI's posterior std
should come out *below* Gibbs's std (the file's own comment, written before
running: "expect close to but below 1" for the VI/Gibbs std ratio).

**Active-learning hypothesis, committed on 2026-10-21,
before `rq7_posterior_disagreement.py` was run:**

> Posterior-disagreement sampling's win rate against plain entropy, across
> 30 trials, will fall in the 50-65% range, similar in magnitude to RQ6's
> restart-disagreement (56.7%) and bootstrap-disagreement (60.0%), since
> this is mechanistically the same "disagreement across a model-uncertainty
> ensemble" idea with a principled posterior standing in for an ad hoc
> ensemble. **Wrong if the win rate is ≤50% (no better than chance against
> entropy) or if it's dramatically higher (>75%), which would suggest it's
> doing something qualitatively different from RQ6's ensembles, not finishing
> the same story.**

Named risk, written down before running: because uncertainty was scoped to
μ only (π, Σ fixed), the signal might end up nearly redundant with plain
entropy instead of genuinely distinct. RQ6's ensembles got real power from
perturbing *all* parameters, including the near-singular collapses this
design can't see at all.

## 3. Experimental setup

**Scoping (decided before writing any code):** π and Σ held fixed
at a current point estimate; only μ₁, μ₂ treated as unknown, with
independent Gaussian priors N(**0**, 10·I) on each (deliberately wide/
uninformative).

**Gibbs sampler** (`rq7_gibbs_sampler.py`): each iteration, (1) sample
every point's hard cluster assignment z_n from the categorical
responsibility distribution under the current μ's (a real random draw, not
the soft r_nk used everywhere else in this project), then (2) sample new
μ_1, μ_2 from the exact conjugate-Gaussian posterior given that Z (derived
and numerically verified against brute-force grid search
before this file was written). 5000 iterations, 1000 discarded as burn-in.

**VI / CAVI** (`rq7_vi_meanfield.py`): each iteration, (1) compute soft
responsibilities r_nk, which is EM's own E-step formula plus a term
`-0.5·tr(Σ_k⁻¹S_k)` that discounts a point's confidence in cluster k by how
uncertain the current belief about μ_k is, then (2) update q(μ_k) =
N(m_k, S_k) via the same conjugate formula the Gibbs sampler uses, but with
soft (fractional) counts r_nk instead of a hard 0/1 assignment. 100
iterations (converges in ~10, per a direct check). ELBO tracked and
confirmed non-decreasing every iteration when unclamped (same guarantee as
EM's log-likelihood, lessons/02; disabled when points are clamped, same
convention as `gmmlab.em.fit`'s own `NonMonotonicLikelihoodError` check).

**Validation datasets:** c=4.0 (well-separated, Z essentially certain,
N=600) for the "does it land near EM's point estimate" check; c=2.5
(N=600, ~200/600 points with soft responsibility between 0.1 and 0.9,
checked directly) for the "does the full posterior differ from a naive
fixed-Z posterior, and do VI and Gibbs agree with each other" checks.

**Active-learning experiment** (`rq7_posterior_disagreement.py`):
reimplements RQ3's query loop, same reason RQ6 did: the new strategy needs
per-step machinery that doesn't fit RQ3's existing branches). At each step:
run the usual warm-started clamped-EM refit to get that step's (π, μ, Σ)
point estimate and the held-out accuracy number; then, holding that step's
π and Σ fixed, run CAVI fresh (clamped points held hard via
`gmmlab.em._apply_clamping`, reused directly rather than reimplemented) to
get q(μ) conditioned on the currently-revealed labels; draw 75 samples of μ
from q(μ) (cheap, since q(μ) is already a plain Gaussian); score every candidate
by the variance of its responsibility across those 75 samples; query the
highest-variance point. Same c=1.0, N=600, budget=60, 30 trials as RQ3-RQ6,
for direct comparability.

## 4. Independent variable(s)

For validation: separation c ∈ {4.0 (well-separated), 2.5 (ambiguous)},
not swept, two deliberately chosen settings. For the active-learning
experiment: strategy ∈ {posterior-disagreement, entropy}, c=1.0 fixed
(matching RQ3-RQ6).

## 5. Dependent variable(s) / metric

Validation: posterior mean and std of μ (both methods), compared to EM's
point estimate and to each other. Active-learning: held-out accuracy on
still-unlabeled points, same convention as RQ3-RQ6; win/loss at the final
budget.

## 6. Controls

- Common random numbers: both strategies in the active-learning experiment
  run on the same 30 datasets, same initial fit, same seeds-per-trial
  scheme as RQ3/RQ6.
- `_apply_clamping` reused directly from `gmmlab.em`, not reimplemented, so
  clamped points are treated identically everywhere in the project.
- Posterior-disagreement's CAVI runs reuse `rq7_vi_meanfield.run_cavi`
  directly (imported, not copy-pasted), so the active-learning experiment
  is built on the exact machinery already validated in Section 7(a)/(b) below.
- Ensemble size for posterior sampling (75 draws from q(μ)) held fixed, not
  swept, same practice as RQ6's fixed ensemble size of 5.

## 7. Results

**(a) Gibbs sampler vs. EM's point estimate, c=4.0** (full output:
`experiments/rq7_gibbs_sampler.py`):

| Quantity | Value |
|---|---|
| EM μ | [[1.917, 0.037], [-2.069, -0.028]] |
| Gibbs posterior mean (5000 iters, 1000 burn-in) | [[1.919, 0.041], [-2.069, -0.028]] |
| Max abs diff (Gibbs mean vs. EM) | 0.0032 |
| Gibbs posterior std (per dim) | ~0.055-0.061 |
| Expected σ/√N (textbook, N=300) | 0.0577 |

**(b) VI vs. EM's point estimate, c=4.0** (full output:
`experiments/rq7_vi_meanfield.py`):

| Quantity | Value |
|---|---|
| VI posterior mean (100 CAVI iters) | [[1.910, 0.037], [-2.075, -0.027]] |
| Max abs diff (VI mean vs. EM) | 0.0078 |
| VI posterior std (per dim) | ~0.054-0.057 |
| ELBO | monotonic increase confirmed over all 100 iterations; first=-2069.522, last=-2065.201 |

**(c) Gibbs's own posterior, fixed-Z vs. full (Z resampled), c=2.5**
(full output: `experiments/rq7_z_uncertainty.py`):

| Quantity | Value |
|---|---|
| Points with genuinely ambiguous responsibility (0.1-0.9) | 200/600 |
| Naive fixed-Z posterior std (per component, per dim) | ~0.056-0.057 |
| Full Gibbs posterior std (per component, per dim) | ~0.062-0.070 |
| Std ratio, full Gibbs / naive fixed-Z | 1.10-1.23 |

**(d) VI vs. Gibbs, c=2.5** (full output: `experiments/rq7_vi_meanfield.py`):

| Quantity | Value |
|---|---|
| VI posterior mean | [[-1.307, 0.030], [1.166, -0.020]] |
| Gibbs posterior mean | [[-1.296, 0.028], [1.181, -0.021]] |
| Max abs diff in means (VI vs. Gibbs) | 0.0154 |
| VI posterior std | ~0.056-0.057 |
| Gibbs posterior std | ~0.062-0.070 |
| Std ratio, VI / Gibbs | 0.81-0.90 |

**(e) Active-learning experiment: posterior-disagreement vs. entropy, c=1.0**
(full output: `experiments/rq7_posterior_disagreement.py`):

| Quantity | Value |
|---|---|
| Posterior-disagreement beats entropy | 24/30 = 80.0% |
| RQ6's restart-disagreement, for comparison | 56.7% |
| RQ6c's bootstrap-disagreement, for comparison | 60.0% |
| Predicted | 50-65% |
| Pre-committed "wrong" line | ≤50% or >75% |
| **Verdict** | **Hypothesis wrong** (80.0% is above the 75% ceiling) |

**(f) Side finding, discovered while picking a validation seed for (a)/(b):**
at c=1.0 (the project's usual separation), scanned 20 data seeds through
the same `fit_best_of` call RQ3-RQ6 use, and every single one converged to an
imbalanced initial fit (pi.min() between 0.027 and 0.42) with accuracy
against true labels between 0.488 and 0.638, i.e. consistently near chance,
never a clean recovery of the true 50/50 structure.

## 8. Unexpected results

Three things, not one:

1. **The c=1.0 collapse is the norm, not a rare pathology.** Previously
   documented (README's "Known gaps," RQ1) as something that *can* happen;
   quantified here for the first time as happening on literally every one
   of 20 scanned seeds. Not a new bug (RQ3-RQ6's strategies are compared
   starting from that same shared point, so their head-to-head results are
   unaffected), but a much stronger statement about the regime than was
   previously on record.

2. **VI's variance underestimation, while theoretically predicted, is worth
   flagging as directly confirmed rather than assumed.** Both the
   VI-vs-Gibbs ratio (0.81-0.90) and the fixed-Z-vs-full-Gibbs ratio
   (1.10-1.23, the opposite comparison) point the same way: the naive
   (fixed-Z, or mean-field) approximations are narrower than the posterior
   that actually accounts for Z's own uncertainty.

3. **The headline surprise: posterior-disagreement's 80.0% win rate is
   well above the pre-registered ceiling, and well above both of RQ6's ad
   hoc ensembles.** The named risk going in was the *opposite* direction:
   that scoping to μ-only uncertainty would make the signal weak or
   redundant with entropy. Instead it produced the single largest margin
   of any disagreement-based strategy tried so far.

## 9. Interpretation

For (1): c=1.0 means the two generating clusters are only one standard
deviation apart, which is heavily overlapping. A full-covariance GMM's
log-likelihood is unbounded above (Lesson 2, Section 2.6, first exploited in RQ1):
a component can always improve raw log-likelihood by shrinking onto a
small, tight sub-cluster rather than finding the true, diffuse 50/50 split.
At this much overlap, that collapsed alternative apparently *beats* the
honest fit on log-likelihood for nearly every random restart, not just an
unlucky few. `fit_best_of`'s `min_weight=0.02` filter catches total
collapse but not a 3-42% minority component, which is this pathology's
actual range here.

For (2): this is the textbook mean-field VI story, now with real numbers
instead of a citation. The true posterior over (Z, μ) is genuinely
entangled: which cluster an ambiguous point belongs to and where that
cluster's mean sits are not independent questions. Mean-field VI's
factorized form q(Z)q(μ) cannot represent that entanglement by
construction, so the single best-fitting *independent* approximation ends
up narrower than the truth. MCMC has no such restriction and, given enough
samples, converges to the actual (wider) posterior.

For (3), the best honest explanation, **not yet independently verified**:
RQ6's ensembles perturb every parameter at once (different restarts or
resampled data refit π, μ, and Σ jointly), so a meaningful fraction of the
disagreement they measure is incidental: different Σ shapes, different
near-singular collapses, residual label-switching jitter after matching,
not all of it tied to genuine μ-uncertainty specifically. Posterior-
disagreement, by construction, measures *only* "how much does this point's
class depend on μ," with no contamination from those other noise sources.
If that is the real mechanism, it is a cleaner, more targeted signal than
RQ6's ensembles rather than a noisier proxy for the same thing, which
would plausibly explain a *larger* margin, not merely a similar one. This
is a mechanism-consistent story, exactly analogous to how RQ6c's "fragility
as targeted signal" explanation was flagged as plausible but unverified
until directly checked. The equivalent direct check here (whether
posterior-disagreement's queried points differ substantially from, or
largely overlap with, entropy's own highest-entropy points) was proposed
but not run, by explicit decision, in favor of documenting this as a
verified result with the mechanism left as an open question.

## 10. Limitations

- The Bayesian treatment is scoped to μ only; π and Σ are still point
  estimates throughout, including inside the active-learning loop (re-fit
  each step by ordinary clamped EM, then held fixed for that step's CAVI
  run). A fuller Bayesian treatment of all three parameters is a
  substantially larger undertaking, deliberately out of scope here.
- Section 9's explanation for the 80.0% result is a plausible, mechanism-
  consistent story, not an independently verified one. No check was run
  on whether posterior-disagreement's queried points are a genuinely
  different set from entropy's, or on whether they coincide with any
  particular structural feature of the data (the kind of direct check
  RQ6/RQ6c's own diagnostics ran for their analogous surprises).
  Documented honestly as open per explicit decision, not oversight.
- Posterior sample count (75 draws from q(μ) per query step) and Gibbs
  iteration count (5000, 1000 burn-in) were fixed choices, not swept or
  tuned.
- The c=1.0 collapse finding (Sections 7f and 8.1) was discovered as a side effect of
  picking a validation seed, on 20 seeds with one restart-seed offset each
  and not a systematic study of how the collapse rate depends on c, unlike
  RQ2's dedicated 140-trial sweep for the initialization-sensitivity
  question. Treat the "every single seed" statement as true for the
  specific setup checked (c=1.0, N=600, 10 restarts, `fit_best_of` defaults),
  not as a universal claim.
- VI's and Gibbs's priors (N(**0**, 10·I)) were chosen once as "wide/
  uninformative" and not varied; a tighter or more informative prior could
  change both the absolute posterior widths and the VI/Gibbs gap.

## 11. Next question

**Primary:** the direct check deferred in Sections 9 and 10. Does posterior-
disagreement query a genuinely different set of points than entropy would,
or mostly the same points in a different order? This is the natural first
thing to check before trusting the "cleaner signal than RQ6's ensembles"
story any further.

**Secondary:** does posterior-disagreement's margin hold up if the
ensemble-size knob (75 posterior samples) is reduced sharply (e.g., to 5,
matching RQ6's ensemble size exactly), to see if the gain is from the
*principled* posterior, or partly just from using a much larger ensemble
than RQ6 did?

**Tertiary, broader scope:** what would it take to extend this to a
genuine posterior over all three parameters (π, μ, Σ), not just μ? Would
that recover something closer to RQ6's ad hoc ensembles' behavior (if the
μ-only scoping really is what makes this signal "cleaner"), or would it
behave differently again?

## 12. Statistical check and replication (added 2026-10-21)

The original 24/30 was significant on its own (exact two-sided binomial test, p = 0.001; 95% interval 0.61 to 0.92), but it was the most surprising result in the project, so it went into the replication too. I reran the comparison unchanged, through the experiment's own `run_trial`, on 100 new trials whose seeds don't overlap the originals (`experiments/rq_replication.py`; per-trial results in `data/replication_results.json`). The criterion was written down before the rerun: an effect counts as replicated only if the new win rate lands on the same side of 50% as the original and a two-sided sign test on the new trials (ties dropped) gives p < 0.05.

**Result:** posterior disagreement won 59, lost 38, and tied 3 (sign test p = 0.04). The average accuracy gap was 3.6 points in its favour.

**What this changes:** replicated, but at a much smaller size. 59% falls inside the 50 to 65% range I originally predicted, and below restart disagreement's 67% in the same replication (RQ6, Section 13). The original 80% was most likely a small-sample overestimate; a 30-trial win rate has a 95% interval roughly 15 points wide either side. The "cleaner signal" explanation in Section 9 was an attempt to explain why posterior disagreement beat RQ6's ensembles so clearly. With the replicated numbers it doesn't beat them, so I've withdrawn that explanation. The primary question in Section 11 (does posterior disagreement pick different points than entropy?) is still worth asking, for a more modest reason: to understand why a real but moderate effect exists.
