# RQ5b: Does TRUE class imbalance (not just the model's belief) change where uncertainty sampling fails?

- **Date started / run:** 2026-10-15
- **Status:** done. Hypothesis held, far more strongly than predicted.
- **Statistical check (added 2026-10-21):** Significant at 30 trials; see Section 12.
- **Code:** `experiments/rq5b_true_imbalance.py` (plus
  `gmmlab.datasets.make_two_gaussians` extended to support an unequal
  split, tested; 74 tests passing) · **Figure:**
  `figures/rq5b_true_imbalance.png`
- **Builds directly on:** `research_log/RQ5_blind_spot.md` (the first RQ5
  angle, the model's *belief* being imbalanced, i.e. a lopsided fit, not
  the data itself) and `research_log/RQ3_entropy_vs_random.md` (the
  uncertainty-vs-random comparison and win-rate this reuses directly).

## Plain-English summary (read this first; Sections 1–6 below are the technical record)

RQ5 (the first pass) tested what happens when the *model's belief* ends up
imbalanced (a lopsided fit) even though the true data is balanced. This
asks a different question: what if the *data itself* is
imbalanced, a true 90/10 split between the two clusters, not 50/50?

A real trap was caught before designing anything: plain accuracy is
misleading under imbalance. A model that always guesses "majority," having
learned nothing at all, would score 90% on a 90/10 split, looking
decent while completely failing to learn the minority cluster. **Balanced
accuracy** (average the accuracy on each class separately, then average
those two numbers) fixes this: the same do-nothing model scores exactly
50% (chance level), correctly exposing that it learned nothing.

The specific worry motivating this test (my own reasoning at the time,
condensed): uncertainty sampling might spend its budget on
majority-dominated boundary points rather than ever discovering the
minority cluster's true extent, a different, more specific failure than
RQ3's general redundancy problem.

**The hypothesis held, far more strongly than predicted.** Uncertainty
sampling beat random sampling in only 2 of 30 trials (6.7%), not just
below the predicted 30-40% range, below it entirely. True imbalance hurts
uncertainty sampling much harder than the general redundancy problem alone
(RQ3's 43.3%) would suggest.

## 1. Research question

(Scoped sub-question of the locked RQ5.) Under a true 90/10 class
imbalance, does uncertainty sampling still lose to random sampling the way
it did under balanced truth (RQ3: 43.3%), and if so, by a similar margin,
or worse?

## 2. Hypothesis

**Committed on 2026-10-15, before any code was written:**

> Random sampling beats uncertainty sampling (balanced accuracy at the same
> budget) in 60-70% of 30 trials under a 90/10 true imbalance, i.e.
> uncertainty sampling wins in 30-40% of trials, noticeably worse than its
> 43.3% win rate under balanced truth (RQ3). **Wrong if uncertainty sampling
> wins in more than 40% of trials** (i.e. if imbalance doesn't clearly hurt
> it beyond RQ3's baseline).

My reasoning at the time, condensed: the specific risk under
imbalance is that uncertainty sampling's queries end up dominated by the
numerically-larger majority class near the boundary, rather than ever
properly locating the minority cluster, a cost beyond the general
redundancy problem RQ3 already found.

## 3. Experimental setup

Same $c=1.0$, $N=600$, initial `fit_best_of` ($R=10$, $n\_iter=2000$),
warm-started refits, query budget $T=60$, and the exact `run_query_loop`
comparison structure as RQ3, the only two things that change:

- **Data:** `make_two_gaussians(c=1.0, sigma=1.0, n_per_component=(540, 60),
  rng=...)`, the library function extended for this (`tests/test_datasets.py`
  has two new tests confirming the split and its validation; 74 tests
  passing, no existing behavior changed when a single int is still passed).
- **Metric:** **balanced accuracy** on the still-unlabeled points:
  $\frac{1}{2}\left(\text{accuracy on majority-class remaining points} +
  \text{accuracy on minority-class remaining points}\right)$, in place of
  RQ3's plain accuracy, for the reason in the summary above. Implemented as
  a new query-loop variant (copies RQ3's mechanics, swaps the metric line)
  rather than overloading `rq3.run_query_loop` with another concern.

30 trials, fresh seeds (not reusing RQ3's, since the data-generating
process itself is different now; reusing RQ3's seeds would not reproduce
anything comparable, unlike RQ4/RQ5's reuse of RQ3's *exact* setup).

## 4. Independent variable(s)

Strategy $\in\{$uncertainty, random$\}$. True split fixed at 90/10,
$c=1.0$, $T=60$ (not swept here: one clear imbalance ratio, not a sweep
across ratios).

## 5. Dependent variable(s) / metric

**Balanced accuracy** on the still-unlabeled points, recomputed after each
query. Verified against the trivial-baseline check in the summary
(always-majority scores exactly 50%, not 90%) before trusting it in the
real run.

## 6. Controls

- **Common random numbers:** both strategies run on the same 30 datasets,
  same initial fit each time, isolating the query strategy as the only
  difference, same logic as RQ3.
- $N=600$, $K=2$, $\sigma=1$, `reg_covar`, EM's `tol` held fixed.
- Random sampling still needs its own sub-repeats (5, matching RQ3) since
  its choices are randomized; uncertainty remains deterministic.

## 7. Results

Full output: `experiments/rq5b_true_imbalance.py`; figure:
`figures/rq5b_true_imbalance.png`.

| Quantity | Value |
|---|---|
| Uncertainty sampling beats random sampling | 2/30 = 6.7% |
| RQ3's win rate, for comparison (balanced truth) | 43.3% |
| Predicted | 30-40% |
| Pre-committed "wrong" line | more than 40% |
| **Verdict** | **Hypothesis held** (6.7% $\ll$ 40%, and below the entire predicted range) |

A real numerical warning showed up during the run
(`RuntimeWarning: divide by zero encountered in log`, from `pi[k]`
underflowing to exactly $0.0$ in floating point during some intermediate
iterations), checked before trusting the headline number: re-ran trials
0-2 individually and confirmed every final `(pi, mu, Sigma)` and every final
accuracy value is finite, not NaN. EM recovers on later iterations before
returning; the warning reflects a transient state under severe imbalance,
not a corrupted result. Not fixed with a pi floor here (it doesn't change
the already-finite outcome), flagged plainly rather than silently
ignored, and worth a floor if it ever starts producing non-finite results
in a future run.

## 8. Unexpected results

**Not a confirmation: an amplification.** 6.7% isn't just "worse than
RQ3's 43.3%," it's below the entire 30-40% range predicted in advance. True
imbalance isn't adding a modest, separate cost on top of the general
redundancy problem; it looks like it's making uncertainty sampling fail
far more completely.

## 9. Interpretation

The transient $\pi\to0$ warning (Section 7) is itself a clue, not just a numerical
footnote: it suggests the fitted model, under this true 90/10 imbalance,
can collapse toward believing the minority cluster is smaller than even the
true 10%, potentially far smaller, approaching the "erase the minority
component almost entirely" extreme. This is RQ1's collapse pathology
again, but now it has something real to latch onto: instead of inventing a
spurious imbalance out of a genuinely balanced truth (RQ1), it can
*exaggerate* a real one.

That connects directly to RQ5(a)'s blind spot (Lesson 4, Section 4.4). If the
model's own belief about the minority cluster shrinks toward
near-nothing, then genuine minority points stop looking ambiguous to the
model at all; they get confidently (and wrongly) folded into the majority
component, exactly the "confidently wrong, therefore invisible to entropy"
situation RQ5(a) already found costly on its own. Uncertainty sampling
never queries them, because they don't look uncertain under the model's own
(collapsing) belief.

Random sampling has no such blind spot: every query has a flat 10% chance
of landing on a genuine minority point, *regardless* of what the model
currently believes. That guaranteed, belief-independent exposure to the
minority class is exactly what uncertainty sampling's rule structurally
lacks, and it's why random wins so lopsidedly here, more lopsidedly than
RQ3's general redundancy story alone would predict.

## 10. Limitations

- One fixed ratio (90/10), not a sweep across imbalance severity: milder
  imbalance (e.g. 70/30) might show a smaller effect; this doesn't show
  where the effect starts appearing.
- The Section 9 mechanism (model belief collapsing toward an even more extreme
  imbalance than the truth) is inferred from the transient warning and the
  RQ1/RQ5(a) connection, not measured directly here (e.g. by tracking `pi`
  over the query loop the way RQ5(a)'s diagnostic did). A direct trace
  would make this more than a plausible story.
- 30 trials, one comparison each at the final budget: a large, clear
  effect, not a high-precision estimate of exactly how large.

## 11. Next question

Does a direct `pi`-trace diagnostic (same as RQ5(a)'s) on one severely
affected trial confirm the model's belief about the minority cluster
actually collapses below the true 10% during the query loop, the way Section 9
infers? That would upgrade this from "a plausible story tying RQ1/RQ3/RQ5(a)
together" to "a mechanism checked directly," the same standard the earlier
RQs were held to.

## 12. Statistical check (added 2026-10-21)

Checked after the fact with an exact two-sided binomial test: 2/30 gives p < 0.001 (95% interval for the win rate 0.01 to 0.22), so this result was already clearly significant at 30 trials and wasn't part of the 100-trial replication.
