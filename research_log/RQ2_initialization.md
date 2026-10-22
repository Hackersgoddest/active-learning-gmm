# RQ2: How sensitive is EM to initialization, and how does that interact with overlap?

- **Date started / run:** 2026-10-13
- **Status:** first pass complete; Section 11's "Primary" follow-up also run; see Section 12
- **Code:** `experiments/rq2_initialization.py` (first pass) and
  `experiments/rq2b_kmeanspp_mechanism.py` (Section 12 follow-up) · **Figure:**
  `figures/rq2_initialization.png`
- **Builds directly on:** `research_log/RQ1_cluster_overlap.md`, Section 11 ("Primary" next
  question) and my notes on initialization methods
  from 2026-10-13 (random vs. k-means++; the two-failure-modes distinction below).

## Plain-English summary (read this first; Sections 1–11 below are the technical record)

We asked two questions: does RQ1's "confident-but-wrong" pattern happen on
most random datasets at heavy overlap, or was it a fluke? And does giving EM
a smarter starting point (k-means++, instead of picking starting points
totally at random) make it go away? We ran 140 independent attempts to find
out: fresh random data every time, 10 restarts each, both ways of starting.

**First answer: common, not universal.** At $c=1$, the pattern showed up on
15 to 17 out of every 30 tries, depending on how EM was started, roughly
half, not nearly all of them. A control at $c=4$ (clusters far apart) showed
it on **zero** of 20 tries, confirming this is about overlap, not
some general glitch.

**Second answer, the surprise: k-means++ didn't help; if anything, it
made the pattern slightly more common, not less,** in both overlap
conditions tested. The likely reason: k-means++'s whole trick is "start your
next guess far from your last one." The confident-but-wrong pattern's small
group is, by definition, a little cluster of points sitting apart from
everything else. A method that goes looking for far-away points to start
from may be more likely to stumble onto exactly that small group first,
not less.

One more thing worth knowing: the strict, extreme version of our "is this
broken" detector fired on **zero out of 120** overlapping attempts. The
milder version we used instead caught it on roughly half. If we'd only had
the strict detector, we'd have wrongly concluded this basically never
happens.

## 1. Research question

Two related questions, not one, because RQ1 revealed they're
different failure modes:

- **(a) Bad-start sensitivity.** How often does EM land in a clearly worse
  local optimum purely because of where it started (the Stage 3 example:
  both initial means on the same side of the true structure)? Does smarter
  initialization (k-means++) reduce how often this happens, compared to
  picking $K$ random data points (what `gmmlab.em._init_params` does now)?
- **(b) Does RQ1's finding generalize?** At $c\le1$, does EM reliably prefer
  a lopsided, partially-collapsed split over the honest one across *many*
  random datasets and restart seeds, or was the one case studied in RQ1 an
  unlucky draw? Unlike (a), this is **not** expected to be fixable by smarter
  initialization or more restarts, because the lopsided split wasn't a worse
  local optimum EM got stuck in; it was the *better-scoring* one even among
  honest attempts.

## 2. Hypothesis

**(a), already committed (Lesson 1, Q5(c)), in my own words, quoted
from `research_log/RQ1_cluster_overlap.md` Section 2(c):**

> EM will become more sensitive to where it starts: different initializations
> should agree closely at large $c$, but at small $c$ they should reach
> different final solutions (different local optima), so results should vary
> more across restarts as $c$ shrinks.

Partially supported already, two independent ways: the restart-ll-variance
column in RQ1's results table (near-zero for $c\ge3$, large for $c\le2$), and
the two reproducible seed examples already logged in `research_log/IDEAS.md`.
RQ2 (a) tests the second half directly: does k-means++ reduce that variance,
or just relocate where EM lands without reducing *disagreement* between runs?

**(b), committed 2026-10-13, in my own words, lightly organized:**

1. **Most** of the 30 random datasets at $c=1$ should show the lopsided,
   confident-but-wrong pattern, not just a minority, "because at $c=1$ the
   centers are that close, making [points from] the other [cluster]
   unluckily closer to the cluster which did not generate [them]." Before running
   anything, I pinned "most" down as **more than 15 of 30** (over half).
2. k-means++ **may reduce** how often it happens, but I remained
   confident the pattern will **still show up often** even with smarter
   initialization, i.e., initialization quality is expected to matter less
   here than it did for the Stage 3 "bad local optimum" example.

Note on the reasoning behind (1): "points landing closer to the cluster that
didn't generate them" is a correct description of *why $c=1$ has overlap at
all* (it's the Bayes-error idea from the RQ1 prep note), but it's a different
claim from "*EM's chosen solution will be lopsided*." Both can be true
together, but they're not the same statement, worth knowing but not a
correction to the prediction itself, which stands as written above.

## 3. Experimental setup

- **Data:** `gmmlab.datasets.make_two_gaussians`, $N=600$ (300/component),
  same as RQ1, at three separations: $c=0.5$, $c=1.0$ (the two from the
  committed hypothesis), and $c=4.0$ as a **control**: RQ1 already showed
  `degenerate% = 0` there for one seed; here we check that holds across many,
  which is what makes the $c\le1$ result mean something (a pattern that
  happens everywhere wouldn't be about overlap at all).
- **Trials:** 30 independent trials each for $c=0.5$ and $c=1.0$ (matching
  my own number); **10** for the $c=4$ control, which is cheaper, and we
  only need to confirm "stays near zero," not precisely measure a rate that's
  expected to already be near zero.
- Unlike RQ1, each trial uses a **freshly generated, independent dataset**,
  not the same data reused across conditions. RQ1 reused one dataset
  deliberately (isolating the effect of $c$ alone); RQ2's question is
  explicitly "does this happen across *different* random draws," so reusing
  one dataset would answer a different question than the one asked.
- **Two initialization strategies per trial, same data, same 10 restarts'
  worth of attempts:** `init="random"` (what `gmmlab.em.fit` already does)
  and `init="kmeans++"` (new, built alongside this design, see below).
  $R=10$ restarts, $n\_iter=2000$, matching RQ1 exactly.
- **Efficiency note, not a methods decision:** call `fit_many` **once** per
  (trial, init strategy) and derive the winning fit from that single list,
  rather than calling `fit_many` and then `fit_best_of` separately as RQ1's
  script did (which, same seed and same work, silently repeated all 10
  restarts a second time). Harmless at RQ1's 6 conditions; needlessly doubles
  a run that's already $\sim$10x bigger here.

## 4. Independent variable(s)

$c \in \{0.5, 1.0, 4.0\}$ and `init` $\in \{$`random`, `kmeans++`$\}$.

## 5. Dependent variable(s) / metrics

Primary metric is **continuous, not binary**, answering RQ1, Section 11's
"Secondary" next question (replace the binary `is_degenerate` threshold with
something that doesn't only catch one extreme):

- For each trial, the **winning fit's smallest component weight**,
  $\min_k \pi_k$ (0.5 = perfectly balanced; near 0 = the lopsided pattern).
  Reported as the **full distribution** across the 30 (or 10) trials per
  (c, init) combination, not collapsed to one yes/no cutoff.
- For readability alongside the distribution, two round reference
  thresholds, reported as plain counts: how many of the 30 trials have
  $\min_k\pi_k < 0.10$ (a generous "lopsided" reading) and how many have
  $\min_k\pi_k < 0.02$ (`is_degenerate`'s existing, stricter threshold, kept
  for continuity with RQ1).
- **This directly tests hypothesis (1):** "more than 15 of 30" at $c=1$
  translates to: more than half of the 30 trials have $\min_k\pi_k < 0.10$.
- **This directly tests hypothesis (2):** compare the $\min_k\pi_k$
  distribution under `random` vs. `kmeans++` init, same $c$, same 30 datasets.

## 6. Controls

$N$, $K=2$, $\sigma=1$, $R=10$, $n\_iter=2000$, $reg\_covar=10^{-6}$ held
fixed throughout. The **same 30 data seeds** are reused for both init
strategies at a given $c$ (so the random-vs-kmeans++ comparison isn't
confounded by also comparing against a different set of 30 datasets); this is
common random numbers again, same logic as RQ1, Section 3, applied one level up.

## 7. Results

Full output: `experiments/rq2_initialization.py`; figure:
`figures/rq2_initialization.png`.

| $c$ | init | n | median $\min_k\pi_k$ | $<0.10$ (lopsided) | $<0.02$ (degenerate) |
|---|---|---|---|---|---|
| 0.5 | random | 30 | 0.134 | 14/30 | 0/30 |
| 0.5 | kmeans++ | 30 | 0.061 | 19/30 | 0/30 |
| 1.0 | random | 30 | 0.090 | 15/30 | 0/30 |
| 1.0 | kmeans++ | 30 | 0.071 | 17/30 | 0/30 |
| 4.0 (control) | random | 10 | 0.491 | 0/10 | 0/10 |
| 4.0 (control) | kmeans++ | 10 | 0.491 | 0/10 | 0/10 |

**Control holds exactly as expected:** at $c=4$, every one of the 20 trials
(both init strategies) landed near-perfectly balanced ($\min_k\pi_k\approx0.49$),
none lopsided. The pattern is specific to heavy overlap, not a general
artifact of the method or the degeneracy machinery.

**The strict `is_degenerate` threshold ($<0.02$) fired on 0 of 120 overlapping
trials, zero across the entire sweep.** Had this sweep used that threshold
alone (as `fit_best_of`'s default does), RQ2 would have concluded the
lopsided pattern "basically never happens." It happens on roughly half to
two-thirds of trials by the more generous ($<0.10$) reading. This is RQ1, Section 11's "Secondary" next question, now with direct evidence for why it
mattered: the binary threshold was blind to exactly the phenomenon this
whole investigation is about.

## 8. Unexpected results

**Hypothesis (1) (most trials lopsided at $c=1$) is only partly right.** The
pattern is clearly common (far more common than the control's 0%) but
"most" in the strong sense doesn't hold uniformly: 14/30 (47%), 19/30 (63%),
15/30 (exactly half), 17/30 (57%) across the four overlapping conditions. Not
"almost every trial," and not rare either, just mixed, trial to trial,
even at fixed $c$. The figure shows this: at both $c=0.5$ and
$c=1.0$, the dots aren't clustered in one place; there's a cluster near the
bottom (lopsided) and a separate cluster higher up (closer to balanced), for
both init strategies.

**Hypothesis (2) was wrong in its optimistic half, and in an informative
direction.** The prediction was "k-means++ may reduce it." It didn't: the
median $\min_k\pi_k$ was *lower* (more lopsided) under kmeans++ at both
overlapping $c$ values (0.061 vs. 0.134 at $c=0.5$; 0.071 vs. 0.090 at
$c=1.0$), and the lopsided-count was *higher* under kmeans++ in both cases
(19 vs. 14; 17 vs. 15). Small sample (30), not a large effect, but
consistent in direction both times, not a coin flip split one way one time.

## 9. Interpretation

The "most trials" question has a precise, honest answer: **common, not
universal.** Roughly half to two-thirds of attempts at heavy overlap land in
the lopsided regime; the rest land somewhere more balanced, by chance, even
with identical settings. This matches Section 9's RQ1 mechanism: the lopsided
split wins on log-likelihood only because a *specific, local* clump of
points happened to sample close together; whether such a clump exists at
all, and how exploitable it is, depends on the specific random draw. Some
draws don't hand EM that opportunity.

The k-means++ result is the surprise, and here's a plausible mechanism,
**not independently verified in this pass**, flagged plainly as
speculation: k-means++'s whole rule is "prefer picking a next center *far*
from centers already chosen." The lopsided solution's small component is,
by definition, a handful of points sitting apart from the main mass. A
selection rule that actively seeks out far-away points may be *more* likely,
not less, to initialize a center near exactly that kind of small, separated
clump, handing EM a head start toward the lopsided solution,
rather than steering it away. This would mean k-means++ is well-matched to
fixing the Stage 3 failure mode (two means landing on the same true side)
precisely because it does the opposite of what's needed here.

## 10. Limitations

- 30 trials (10 for the control) is enough to see a clear pattern against the
  control, not enough to pin down precise rates: the gap between 14/30 and
  19/30 could narrow or widen with more trials.
- The k-means++ mechanism in Section 9 is a plausible story, not a verified one. A
  direct test (e.g., checking whether kmeans++'s *second* chosen point lands
  near the eventual small component, trial by trial) was not run here.
- $K=2$, one choice of $N$, one choice of restart count; none of it varied.
- This still doesn't establish *how often* a given random dataset contains an
  "exploitable" clump in the first place, only how often EM finds and uses
  one when present.

## 11. Next question

**Primary:** directly test the proposed k-means++ mechanism: does its
second chosen starting point land measurably closer to the eventual small
component's data than a random second point would, across these same
trials? A cheap, direct check of Section 9's speculation rather than leaving it as
a story.

**Secondary:** now that "lopsided but not literally degenerate" is confirmed
common, RQ5 (uncertainty $\ne$ informativeness, already motivated by
RQ1) has a second, independent mechanism to study: not just the extreme
collapse case, but this much more common partial-imbalance case specifically.

## 12. Follow-up: testing the k-means++ mechanism from Sections 9 and 11 directly (2026-10-13)

**Why this is a separate section, not a rewrite of Section 2(b) above.** Section 2(b)'s
hypothesis was committed *before* running RQ2 and is left exactly as
written, vagueness included: editing a hypothesis after already knowing the
result (here, after seeing Section 7's table) would make the record untrustworthy,
not sharper. This section is a new, separate hypothesis, committed
*before* this follow-up ran, about a question Section 11 explicitly left open.

**Hypothesis, committed before running:**

> If k-means++'s second point is chosen by a distance-weighted lottery whose
> outcome depends on where the first (uniformly random) point lands,
> strongly pulled toward the small/lopsided component when the first point is
> in the big component, and pulled toward the big component instead when the
> first point already landed in the small one, then, across trials where the
> winning fit came out lopsided, k-means++'s second point will land inside
> that small component in roughly **80%** of cases (vs. a pure-chance baseline
> around 7-13%, the small component's typical share of the data).
> **Wrong if the real rate is below 60%.**

**Method.** `experiments/rq2b_kmeanspp_mechanism.py`. Reuses RQ2's exact
per-trial data/restart seeds and the $<0.10$ lopsided threshold from Sections 5 and 7,
for $c\in\{0.5,1.0\}$ only ($c=4$ has ~0 lopsided trials, nothing to measure
a mechanism on). `em.py`'s `GMMFit` gained a new field, `init_indices`
(the row indices of $X$ chosen as starting means, in order; `_init_random`
and `_init_kmeanspp` both updated to return them; `tests/test_em.py` updated
for the new 4-tuple return, 68 tests still passing). For each trial whose
winning fit was lopsided: found the small component ($\arg\min_k \pi_k$),
hard-assigned every point by $\arg\max$ responsibility, and checked whether
the data point k-means++ chose **second** was one of the points hard-assigned
to the small component. The chance baseline is computed the same way, per
trial (fraction of all points hard-assigned to the small component), not
taken from Section 7's $\pi$-based median, which turned out to measure something
different (see note below).

**Results (36 lopsided trials found: 19 at $c=0.5$, 17 at $c=1.0$):**

| Quantity | Value |
|---|---|
| k-means++'s 2nd point landed in the small component | 7/36 = 19.4% |
| Mean per-trial chance baseline | 2.4% |
| $c=0.5$ only | 4/19 = 21.1% |
| $c=1.0$ only | 3/17 = 17.6% |
| Predicted | ~80% |
| Pre-committed "wrong" line | below 60% |
| **Verdict** | **Hypothesis wrong** (19.4% < 60%) |

**Unexpected result.** The hypothesis is wrong by its own pre-committed line,
not a close call. But it isn't *nothing*: 19.4% is still roughly **8x** the
2.4% chance baseline, and the direction held at both $c=0.5$ and $c=1.0$
separately (21%, 18%, not one high value carrying the average). So some
real, k-means++-specific effect is present; it's just far weaker and more
indirect than "its second point becomes one of the leftover points."

A separate, incidental finding while building this check: the small
component's $\pi$ weight (Section 7's metric, the *average* soft responsibility
across all 600 points) and its hard-assignment count (points that actually
win $>50\%$ responsibility) disagree substantially: e.g. one trial had
$\pi_{\min}=0.054$ but only 0.8% of points ($\approx5$ of 600) won
the small component outright. Many points carry partial, sub-50% pull toward
the small component without ever being claimed by it. This means Section 7's
reported rates and this section's baseline are measuring different
things, both legitimate, and shouldn't be compared against each
other.

**Interpretation.** The guessed mechanism (k-means++ seeding a
center onto the leftover clump) is at most a minor contributor, not the main
story. A plausible reason the effect is this weak (not verified here,
flagged as speculation): the mechanism assumed the leftover clump is
*spatially far* from the main mass, the same way k-means++'s distance
weighting works. But the collapse pathology (lessons/02, Section 2.6) is about a
*tight*, near-singular clump of points: tightness, not distance from the
center, is what makes it exploitable. A tight clump can sit anywhere,
including close to the overlap region's center, where k-means++'s
far-point-seeking rule wouldn't especially favor it. If true, this also
explains why k-means++ still made the *overall rate* slightly worse in Section 7's
table (it's doing *something* right occasionally) without that something
being "landing on the clump" most of the time.

**Limitations.** 36 lopsided trials is a workable but modest sample:
individual c-level rates (21%, 18%) have real sampling noise. This check only
asks whether k-means++'s second point was *literally inside* the small
component; it doesn't test the softer claim "landed *closer* than a random
point would, even without landing inside it," which could still be true and
would need a distance-based check, not a membership check, to test.

**Next question:** the spatial-vs-tight distinction above is itself a
testable, not-yet-tested claim: e.g., compare each lopsided trial's small
component's *spread* (within-component variance) against its *distance* from
the overall data centroid, across trials, to see whether "tight" and "far"
usually go together or usually don't. Given Oct 20 scope pressure, this is
recorded here as a documented next step, not run now; Stage 7 (the bridge
lesson) is next.
