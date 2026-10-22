# RQ1: How does cluster overlap affect EM's ability to recover latent structure?

- **Date started:** 2026-09-13
- **Design finalized / run:** 2026-10-12
- **Status:** first pass complete; see Section 11 for the follow-up this motivates
- **2026-10-13:** `experiments/rq1_cluster_overlap.py` was computing its 10
  restarts twice per condition (calling both `fit_many` and `fit_best_of`
  with the same seed, since `fit_best_of` redoes the work internally). Fixed to
  call `fit_many` once and derive the winner via `gmmlab.em.best_of`.
  Rerun and checked: every number in Section 7's table is unchanged; runtime
  dropped from ~52s to ~28s, confirming the fix removed exactly the
  duplicated work and nothing else.
- **Code:** `experiments/rq1_cluster_overlap.py`
- **Figures:** `figures/rq1_cluster_overlap.png` (the 4-panel summary trends) and
  `figures/rq1_cluster_overlap_fits.png` (one panel per $c$: the data, the
  true means, and what EM actually recovered, the picture that makes Section 8's
  finding visible directly and not just readable in the table)

## Plain-English summary (read this first; Sections 1–11 below are the technical record)

We generated 600 fake data points: 300 secretly from a "bell" centered at
$+c/2$, 300 from one centered at $-c/2$, then erased the labels and asked EM
to find two groups. We predicted that as the two centers got closer together
(smaller $c$), the model would become more *uncertain* about the points in
between, with entropy rising toward its maximum.

**That prediction was wrong at $c\le1$, in an informative way.** Instead of
becoming more uncertain, the fitted model became *more confident*, exactly
backwards. At $c=1.0$ its confusion score in the tricky middle zone was
`0.000` (totally sure) when it should have been close to the maximum
(`0.684`, using the real, secret centers for comparison).

**Why:** EM isn't trying to rediscover the two true causes; it's trying to
find whatever explanation scores best on the data it actually sees. When the
two true clusters overlap heavily, an honest 50/50 split explains *every*
point only vaguely (low score everywhere). EM can score higher instead by
carving out a small pocket of points that happen to sit close together by
chance, fitting a tiny group snugly around just those, and dumping everyone
else into one big leftover group. That lopsided split (e.g. "97% of points
here, 3% over there") beats the honest split on EM's own
scorecard, even though it isn't the real structure. Because almost every
point obviously isn't in that tiny odd pocket, the model ends up reporting
near-total confidence almost everywhere: confident about the wrong thing.
Checked across all 10 random restarts, not a fluke of one run: 9 of 10 found
this same lopsided pattern.

**Why it matters for the project:** this is a concrete case of a model
reporting high confidence for reasons that have nothing to do with
understanding the data, which is exactly the risk this whole project is about,
found with real numbers instead of assumed.

## 1. Research question

Fixing $K=2$ spherical components ($\boldsymbol{\Sigma}_k = \sigma^2\mathbf{I}$),
how does shrinking the separation
$c = \lVert\boldsymbol{\mu}_1-\boldsymbol{\mu}_2\rVert/\sigma$ affect (a) the
model's assignment uncertainty and (b) EM's ability to recover the true labels?
(Initialization sensitivity is tested systematically in RQ2; see the observation
in Section 2(c).)

## 2. Hypothesis

*(my own words, from Lesson 1 Q5, lightly organized, not rewritten)*

As $c$ decreases from large toward small:

- (a) **Uncertainty.** Points lying between the two components will have
  responsibilities that become more similar to each other, so their assignment
  entropy $\mathbb{H}[z_n\mid\mathbf{x}_n]$ will increase, moving from near $0$
  (confident) toward $\ln K$ (fully uncertain) as $c\to0$.
- (b) **Recoverability.** The ability to recover the true labels will decrease,
  since it becomes harder for EM to correctly identify which component generated
  each point.
- (c) **Initialization sensitivity (observation, tested in RQ2).** EM will
  become more sensitive to where it starts: different initializations should
  agree closely at large $c$, but at small $c$ they should reach different final
  solutions (different local optima), so results should vary more across restarts
  as $c$ shrinks. In RQ1 this is only recorded as an observation (restart
  variance of the final log-likelihood); RQ2 tests it systematically.

## 3. Experimental setup

- **Data generator:** `gmmlab.datasets.make_two_gaussians(c, sigma=1.0, n_per_component=300)`,
  with $K=2$, equal weight, spherical $\boldsymbol\Sigma=\sigma^2\mathbf{I}=\mathbf{I}$,
  means at $(\pm c/2,\,0)$. $N=600$ total per condition (timed: 10 EM restarts
  on this size takes well under a second, so cost is not a constraint here).
- **$c$ grid:** $\{0.5,\ 1,\ 2,\ 3,\ 4,\ 6\}$, the same six values already
  worked by hand in `lessons/RQ1_prep_separation_and_bayes_error.md`, so every
  number this experiment produces can be checked against a number already
  computed there.
- **Restarts per condition:** $R=10$, via `gmmlab.em.fit_many` (every restart
  kept, not just the best, needed for the restart-variance observation in
  Section 2(c)). The **best** restart is selected via `gmmlab.em.fit_best_of`'s
  degeneracy guard (see the finding directly below), not a plain
  highest-log-likelihood pick; all 10 raw results still feed the
  variance-across-restarts observation.
- **$n\_iter=2000$ for this sweep** (not the library default of 100). Found
  while finalizing this design, at $c=2$: with only 100 iterations
  none of the 10 restarts converge at all; with enough iterations to
  converge, EM finds a *higher*-log-likelihood solution that is NOT the true
  2-cluster structure: one component collapses onto 3 of 600 points with a
  covariance eigenvalue of $2\times10^{-5}$, exactly the unbounded-likelihood
  pathology `lessons/02_jensen_and_elbo.md` Section 2.6 names. `gmmlab.em.is_degenerate`
  (a component holding $<2\%$ of the data) now flags this, and
  `fit_best_of` picks the best **non**-degenerate restart. Verified this
  recovers means within $0.5$ of the truth at $c=2$ even with 2000 iterations
  available (`tests/test_em.py::test_unbounded_likelihood_pathology_is_detected_and_avoided`).
  Full numbers in `research_log/IDEAS.md`.
- **Seeding, common random numbers, deliberately:** the SAME master seed
  (`np.random.default_rng(0)` for data, `np.random.default_rng(1)` for the 10
  restart sub-seeds) is reused at **every** $c$. Because `make_two_gaussians`
  only uses $c$ to shift *where* the means sit, not which raw noise values get
  drawn, this means the exact same underlying random points (just relabeled by
  shifted means) and the exact same 10 initialization attempts are reused
  across the whole sweep. This isolates the effect of $c$ alone: a difference
  between two $c$ conditions can't be "that condition just got a luckier
  draw," because both conditions drew from the identical random stream. This
  is a real design choice, not an arbitrary default; see Section 10 for what it
  costs (single master seed, not replicated).
- **One run per $c$**, not independent repeated trials. The common-random-numbers
  design above substitutes for some of what replication would otherwise need;
  what it does NOT give us is an estimate of how much the *whole pattern*
  would shift under a *different* master seed. Flagged plainly in Section 10, not
  fixed here: first pass, not the final word.

**"Central overlap region" (Section 5), now made precise.** By
`lessons/RQ1_prep_separation_and_bayes_error.md`, Section P.3, a point's responsibility
here depends only on its position along the separating axis (the $x$-coordinate,
by construction), not on $y$. Defined as: $\{\,\mathbf{x}_n : |x_n| \le 0.5\sigma\,\}$,
within half a standard deviation of the true midpoint, along $x$ only, fixed
across every $c$. Chosen round and symmetric; not sensitivity-checked against
other thresholds yet (if the conclusion turns out to hinge on this exact
number, that becomes a next question, not a silent assumption).

## 4. Independent variable(s)

$c = \lVert\boldsymbol{\mu}_1-\boldsymbol{\mu}_2\rVert/\sigma$.

## 5. Dependent variable(s) / metrics

- mean assignment entropy over **all** points (global picture)
- mean assignment entropy over points in the **central overlap region**,
  $|x_n|\le0.5\sigma$ (defined precisely in Section 3)
- label-recovery accuracy (best permutation match against ground-truth $z$),
  reported alongside the Bayes-optimal accuracy $1-\Phi(-c/2)$ (`lessons/RQ1_prep_separation_and_bayes_error.md`, P.2);
  the **gap** is the quantity of interest, since even perfect knowledge loses
  accuracy as $c$ shrinks
- observation only, for RQ2: variance of final log-likelihood across random EM
  restarts, per $c$

Note: entropy under the *fitted* model mixes two things, intrinsic ambiguity
(the true distributions really overlap) and fitting error (EM found a poor
model). To separate them, also compute entropy under the *true* parameters as a
reference.

## 6. Controls

$N$, $K$, $D$, $\sigma$, EM convergence tolerance, and the random seed schedule
held fixed across $c$ values so only separation changes.

## 7. Results

Full numbers: `experiments/rq1_cluster_overlap.py` output below; figure:
`figures/rq1_cluster_overlap.png`.

**Reading the columns, plainly:**

- **gap** = EM acc $-$ Bayes acc. A *large* negative gap (e.g. $c=0.5$:
  $-0.100$) means EM is underperforming. A *small* gap near zero,
  either sign (e.g. $c=6$: $-0.002$), usually isn't EM underperforming at all;
  see the note after the table for why, worked out in detail for the $c=6$ case.
- **$\mathbb{H}$ all (fit)** = average confusion over *every* point, which
  blends obviously-easy points with the few tricky ones near the boundary;
  the next two columns separate that out.
- **$\mathbb{H}$ overlap (fit)** = average confusion, but only for points near
  the true boundary, using the model's *own fitted* guess.
- **$\mathbb{H}$ overlap (true)** = the same boundary points, but using the
  real, secret centers instead of the fitted guess: the honest reference
  for how ambiguous those points are. The gap between this column
  and the previous one is Section 8's whole finding.
- **n overlap** = how many of the 600 points fell in that boundary
  region. Shrinks from 223 at $c=0.5$ to just 4 at $c=6$, so the two overlap
  columns at $c=6$ are averaging over only 4 points, a much shakier number
  than the ones at $c=0.5$.
- **restart ll var** = how much the 10 random restarts' *final* scores
  disagreed with each other. Large at $c=0.5$ (restarts land on
  different answers); essentially zero from $c=3$ up (every restart finds the
  same answer regardless of where it started), a preview of RQ2.
- **degenerate%** = of the 10 restarts, the percentage that hit the *extreme*
  version of the collapse ($<2\%$ of the data in one component). This only
  catches the worst cases: $c=1.0$ shows $0\%$ here even though the winning
  restart was still a 97%/3% split, which is the milder version of the exact
  same pathology. This column understates how often the lopsided pattern
  occurred.

| $c$ | EM acc | Bayes acc | gap | $\mathbb{H}$ all (fit) | $\mathbb{H}$ overlap (fit) | $\mathbb{H}$ overlap (true) | n overlap | restart ll var | degenerate% |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.498 | 0.599 | $-0.100$ | 0.140 | 0.146 | 0.691 | 223 | 22.6 | 20% |
| 1.0 | 0.505 | 0.692 | $-0.187$ | 0.063 | **0.000** | 0.684 | 202 | 2.6 | 0% |
| 2.0 | 0.727 | 0.841 | $-0.115$ | 0.280 | 0.305 | 0.655 | 156 | 7.9 | 10% |
| 3.0 | 0.943 | 0.933 | $+0.010$ | 0.154 | 0.548 | 0.608 | 85 | $8\times10^{-14}$ | 0% |
| 4.0 | 0.978 | 0.977 | $+0.001$ | 0.049 | 0.533 | 0.555 | 29 | $5\times10^{-14}$ | 0% |
| 6.0 | 0.997 | 0.999 | $-0.002$ | 0.004 | 0.532 | 0.529 | 4 | $8\times10^{-16}$ | 0% |

(Hypothesis (b)'s prediction (accuracy drops as $c$ shrinks) holds, and the
gap to Bayes-optimal is near-zero from $c=3$ up, confirming EM is basically
recovering everything recoverable there. The small gaps at $c=3,4,6$ switch
sign ($+0.010$, $+0.001$, $-0.002$) and that's expected, not noise to explain
away each time individually: "Bayes-optimal accuracy" ($1-\Phi(-c/2)$) is a
**population average over infinitely many hypothetical batches**, not a count
achievable on any one real batch of 600; $0.9987\times600=599.22$ isn't even
a whole number. EM's accuracy is a real count on this one batch. Checked
at $c=6$: EM got 598/600 right, and a classifier using the TRUE
means on this exact same batch *also* got 598/600 right, wrong on the
identical same 2 points. EM did not underperform here; it tied the best
achievable score on this specific sample. Which side of zero a small gap
lands on is just which way this one random batch happened to go; a different
seed could flip the sign. This is the single-master-seed limitation (Section 10),
made concrete.)

## 8. Unexpected results

**Hypothesis (a) was wrong, in a specific and informative way, at $c\le1$.**
The prediction was that assignment entropy in the overlap region would
*increase* toward $\ln 2$ as $c$ shrinks. Instead, at $c=1.0$ the fitted
model's entropy there is **exactly 0.000**, maximally *confident*, while
the entropy computed under the TRUE generating parameters, for the identical
points, is $0.684$ (barely below $\ln2=0.693$, i.e. near-maximally
ambiguous). The model isn't uncertain where it should be; it's confidently
wrong. Same pattern, less extreme, at $c=0.5$ ($0.146$ fitted vs. $0.691$
true) and $c=2.0$ ($0.305$ vs. $0.655$).

**Mechanism, found by inspecting the fitted parameters:** at $c=1.0$,
the winning fit has $\boldsymbol\pi=(0.966,\,0.034)$: one component holds
3.4% of the data, with a covariance eigenvalue of $0.042$ (a thin sliver, not
a literal point-collapse, but the same pathology on a smaller scale as the
one `gmmlab.em.is_degenerate` was built to catch at $c=2$ during design). This
wasn't bad luck from one restart: checking the minimum component weight
across **all 10** restarts at $c=1.0$ gives
$(0.023,0.023,0.023,0.023,0.034,0.034,0.034,0.034,0.034,0.436)$, with nine of
ten restarts finding an imbalanced, partially-collapsed solution; only **one**
found anything resembling the true 50/50 split, and that balanced solution
did not have competitive log-likelihood (it didn't win). At $c=0.5$, every
single restart's minimum weight was below $0.132$; not one found balance.
`is_degenerate`'s $2\%$ threshold, calibrated against the extreme collapse
found at $c=2$, does not catch this milder, graded version of the same
unbounded-likelihood pathology (lessons/02, Section 2.6): it's a continuum, not a
binary, and our guard currently only catches one end of it.

## 9. Interpretation

Two separate things were tangled in the original hypothesis, and this run
pulled them apart. "Entropy rises as clusters overlap" is true of the
**true, correctly-specified** model (confirmed: the green TRUE-params curve
in the entropy panel rises smoothly and monotonically toward $\ln2$ as $c$
shrinks, exactly as predicted). It is **not** what the **fitted** model
necessarily shows, because at high overlap EM has an easier, higher-likelihood
option available that the true-parameter curve doesn't: carve off a small,
non-representative sliver of points into one component (inflating its density
there) and let the other component absorb the rest. That solution explains
*this specific finite sample* better than the honest, symmetric 2-cluster
fit does, precisely because with heavy overlap there often isn't enough
genuine separating signal in 600 points for the symmetric truth to win on
likelihood against a solution that's technically freer to overfit a subset.
The result is a model that is *confidently* wrong rather than *honestly*
uncertain: the opposite failure mode from the one hypothesized, and a much
more concrete, concerning instance of "uncertainty $\ne$ informativeness"
than the project's central caveat was stated in the abstract.

## 10. Limitations

- **Single master seed per condition** (documented as a deliberate
  common-random-numbers design in Section 3, not an oversight): this experiment
  shows what happened on *one* realization per $c$, not how often the
  partial-collapse pattern occurs across many datasets at the same $c$. The
  degree of imbalance (e.g. $3.4\%$ vs. $23\%$ vs. perfectly balanced) could
  itself vary a lot run to run; unknown from this pass.
- **The $2\%$ degeneracy threshold is shown, by this experiment, to be too
  lenient** for the graded version of the collapse pathology, a limitation
  of the current guard and not just of this result's interpretation.
- $N=300$ per component, $K=2$ only, one dataset shape (spherical); none
  varied here.
- The $0.5\sigma$ overlap-region width was chosen, not swept; unknown how
  sensitive these numbers are to that specific choice.

## 11. Next question

**Primary:** does the partial-collapse pattern at $c\le1$ hold across many
random datasets, or was this one draw unlucky? Repeat $c\in\{0.5,1\}$ across,
say, 30 independent master seeds (data **and** restarts), and track the
minimum component weight of the winning fit each time. If it's consistently
small, "EM becomes confidently wrong under heavy overlap" holds up as a
general finding and not an artifact of one sample, and it's a natural
RQ6/RQ7 candidate: does an ensemble (bootstrap or multi-restart) disagreement
signal catch what a single fit's own entropy misses here?

**Secondary, methodological:** replace the binary `is_degenerate` threshold
with a continuous measure (e.g. each component's *effective sample size*
$N_k=\sum_n r_{nk}$, or $\pi_k$'s own entropy across components) so "how
collapsed is this fit" is a number, not a yes/no that only catches one
extreme.

**Status update:** promoting the "degenerate solutions beat honest ones"
finding from an implementation detail to the headline result of RQ1: it's
more informative than the originally hypothesized smooth entropy rise, and
it strengthens RQ5's motivation (an entry already added to
`research_log/IDEAS.md`).
