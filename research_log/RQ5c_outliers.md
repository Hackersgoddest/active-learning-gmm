# RQ5c: Does uncertainty sampling get fooled by low-density outliers that look ambiguous but aren't informative?

- **Date started / run:** 2026-10-15
- **Status:** done. Hypothesis held; two real numerical bugs found and
  fixed along the way; see Section 8.
- **Statistical check (added 2026-10-21):** Significant at 30 trials; see Section 12.
- **Code:** `experiments/rq5c_outliers.py` · **Figure:**
  `figures/rq5c_outliers.png`
- **Builds directly on:** `research_log/RQ3_entropy_vs_random.md` (the
  uncertainty-vs-random comparison this reuses) and the conversation
  working out, step by step, exactly where a misleading outlier would have
  to sit in this specific data (see Section 3).

## Plain-English summary (read this first; Sections 1–6 below are the technical record)

A genuine boundary point is ambiguous because it fits *both* clusters'
stories reasonably well. An outlier is different: it's not caught between
two good explanations, it's just badly explained by *either*. The question:
can a point be constructed that *looks* exactly as ambiguous as a real
boundary point (same entropy) while sitting somewhere rare
and uninformative, and if uncertainty sampling can't tell the
difference, does it waste budget on it (and even actively distort the fit)
more than random sampling does?

Worked out from the data's own structure (the two true means
differ *only* along x; the y-axis carries no information distinguishing
the clusters at all, same spread in every direction): entropy here depends
*only* on a point's x-coordinate. So a point at $x\approx0$ (looks exactly
as ambiguous as any real boundary point) with an extreme $y$ (rare: $y=8$
is far beyond the conventional 3-standard-deviation "this
basically never happens" line) is a disguised outlier: high entropy,
low density, and, because clamping a point with an extreme y can distort
the covariance estimate (Lesson 4, Section 4.2's A/B/C mechanism), not just waste a
query, potentially harmful to query.

**The hypothesis held.** Uncertainty sampling beat random sampling in only
6/30 trials (20.0%), well below RQ3's 43.3% no-decoy baseline and below
the 40% "wrong" line. Confirmed, not just inferred: uncertainty
sampling queried decoys at **4.5x** the rate pure chance would (2.23 vs.
0.50 per 60 queries); random sampling's rate (0.47) landed almost exactly
on the chance baseline, as it should.

Getting there took two real debugging detours, both the same underlying
mistake at different scales: placing the decoys too close together let EM
collapse a near-singular covariance around them as a *group* (an even more
extreme version of RQ1's unbounded-likelihood pathology than intended),
twice tripping `NonMonotonicLikelihoodError` from genuine floating-point
precision loss, not a logic bug. Fixed by spreading the 5 decoys apart from
*each other* (not just jittering them as a tight unit), stress-tested
across all 30 trial seeds before trusting the real run. Full account in Section 8.

## 1. Research question

(Scoped sub-question of the locked RQ5.) Does uncertainty sampling, unable
to distinguish a disguised low-density outlier from a genuine boundary
point (both have equally high entropy here), lose to random sampling by
*more* than RQ3's general redundancy baseline (43.3% win rate)?

## 2. Hypothesis

**Committed on 2026-10-15, before any code was written:**

> Uncertainty sampling's win rate against random sampling will be lower
> than RQ3's 43.3% baseline. **Wrong if uncertainty sampling's win rate
> exceeds 40%.**

## 3. Experimental setup

Same $c=1.0$, $N=600$ real points, initial `fit_best_of`, warm-started
refits, budget $T=60$, plain accuracy (no imbalance here, unlike RQ5b, the
standard metric is fine), 30 trials, as RQ3.

**New piece: 5 injected decoy points per trial**, appended to the 600 real
ones (605 total points available to query):

- $x \approx 0$ (exactly 0.0), ties the maximum possible entropy, the
  same ambiguity level as a genuine boundary point.
- $y = 8$, roughly 8 standard deviations out; the real data generator
  essentially never produces this.
- True label assigned arbitrarily (half 0, half 1); $x=0$ doesn't favor
  either true cluster anyway, so this is a deliberate bookkeeping choice,
  not something nature determines.

**Held-out accuracy is computed over the 600 real points only**: the
decoys are a distraction being measured for their *effect*, not part of
the structure a strategy is being graded on recovering. Both
strategies CAN query a decoy (it's eligible, same as any other point); it
just doesn't count toward the accuracy score either way.

**Additional diagnostic (not part of the hypothesis test, same role as RQ3, Section 7's lopsided/balanced breakdown):** fraction of each strategy's 60 queries
that landed on a decoy, compared to the base rate a point would be queried
by pure chance ($5/605 \approx 0.8\%$), to check the mechanism,
not just the outcome.

## 4. Independent variable(s)

Strategy $\in\{$uncertainty, random$\}$. $c=1.0$, $T=60$, 5 decoys at
$(x,y)=(0,8)$-ish held fixed (not swept).

## 5. Dependent variable(s) / metric

Plain accuracy on the still-unlabeled REAL points only (decoys excluded
from the metric, included in what's queryable). Same win/loss framing as
RQ3: compare final accuracy at the full budget.

## 6. Controls

- **Common random numbers:** both strategies run on the same 30
  datasets (600 real points + 5 decoys each), same initial fit.
- $N=600$ real points, $K=2$, $\sigma=1$, `reg_covar`, EM's `tol` held
  fixed, matching RQ1-RQ5.
- Random sampling still needs 5 sub-repeats (randomized choices);
  uncertainty remains deterministic.

## 7. Results

Full output: `experiments/rq5c_outliers.py`; figure:
`figures/rq5c_outliers.png`.

| Quantity | Value |
|---|---|
| Uncertainty sampling beats random sampling | 6/30 = 20.0% |
| RQ3's win rate, for comparison (no decoys) | 43.3% |
| Predicted | lower than 43.3% |
| Pre-committed "wrong" line | exceeds 40% |
| **Verdict** | **Hypothesis held** (20.0% $\ll$ 40%) |
| Mean decoy queries/60, uncertainty sampling | 2.23 |
| Mean decoy queries/60, random sampling | 0.47 |
| Pure-chance baseline | 0.50 |

## 8. Unexpected results

**Two real numerical bugs, not part of the research question itself, but
worth recording in full since they're real findings about the method.**

*First:* the initial version placed all 5 decoys at *identical*
coordinates $(0, 8)$, a true zero-variance cluster. The very first full
run threw `NonMonotonicLikelihoodError` (a real log-likelihood decrease of
$3.880\times10^{-3}$). Traced by checking whether the error persisted
without the decoys (it didn't) and then checking each of `fit_best_of`'s
10 restarts individually: one restart's `pi` landed on exactly $5/605$
(0.00826) with that component's covariance eigenvalue sitting exactly at
the `reg_covar` floor: EM had found and collapsed onto the literal
duplicate-point cluster, and the iteration path through that near-singular
configuration lost enough floating-point precision to trip the check.
Fixed with a small shared jitter (std=0.05).

*Second:* that fix wasn't enough. A full run with the jittered version
still threw the same error on a different trial, a much smaller decrease
this time ($2.533\times10^{-8}$, barely past the $-10^{-8}$ tolerance).
Same root cause at a smaller scale: std=0.05 still kept the 5 decoys in a
tight enough sub-cluster that EM could fit a near-singular covariance
around all of them as a group. Fixed properly this time by spreading the 5
decoys apart from *each other* ($x$: std=0.1, $y$: std=1.0 around
`DECOY_Y`), so no single covariance can cover all of them tightly.
Stress-tested across all 30 trial seeds (initial fit only, then the full
query loop on 5 trials) before trusting the real run.

**On the research question:** the hypothesis held cleanly, and the
decoy-query-rate diagnostic confirms the mechanism, not just the
outcome: uncertainty sampling does over-select these worthless
points (4.5x chance), exactly as the entropy-only-depends-on-x reasoning
predicted.

## 9. Interpretation

Entropy, in this data, depends only on how far a point sits from the
midpoint along the one axis that actually distinguishes the two clusters.
It has no way to separate "ambiguous because it's a genuine boundary case"
from "ambiguous because it's nowhere near either cluster's typical
territory, just coincidentally equidistant." A strategy built entirely on
entropy inherits that blindness. Random sampling doesn't have this problem
not because it's smarter, but because it isn't looking at entropy at all:
it treats a decoy exactly like any other point, which is this one case
where that's the right call.

The two numerical bugs are, in a loose sense, the same story one level
down: EM itself can't tell "a real, honest cluster" from "a tight group of
points that happen to sit together," and will fit a near-singular
covariance to either given the chance (RQ1's original finding). The decoys
were designed to fool the *query strategy*; it turned out, before that,
they could also fool *EM's own fitting* if placed carelessly.

## 10. Limitations

- One fixed decoy count (5) and placement ($y=8$, spread std=1.0), not a
  sweep. Says nothing yet about milder outliers (e.g. $y=4$, closer to the
  3-std line) or about whether the effect scales with more decoys.
- The decoy-query-rate diagnostic (Section 7) is a strong, direct confirmation of
  the mechanism, but the *downstream* cost (how much the resulting
  accuracy actually suffers once a decoy gets clamped and distorts the
  fit, per Section 1's prediction) wasn't measured separately from the overall
  win/loss count.
- Doesn't test whether a strategy that combines entropy with a density
  check (e.g., only query points that are both ambiguous *and* not
  themselves low-density) would avoid this failure.

## 11. Next question

Does a density-aware variant of uncertainty sampling (e.g., entropy among
only the points with reasonably typical local density, excluding anything
too far into a sparse region) recover most of the lost win rate here,
confirming the fix is as simple as "also check density," or is the problem
harder than that once combined with the redundancy (RQ3/RQ4) and blind-spot
(RQ5a) failures already found?

## 12. Statistical check (added 2026-10-21)

Checked after the fact with an exact two-sided binomial test: 6/30 gives p = 0.001 (95% interval for the win rate 0.08 to 0.39), so this result was already clearly significant at 30 trials and wasn't part of the 100-trial replication.
