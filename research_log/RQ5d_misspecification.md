# RQ5d: What does model misspecification (wrong K) cost, and does it change the uncertainty-vs-random comparison?

- **Date started / run:** 2026-10-17
- **Status:** done. Part (a) of the hypothesis held almost exactly; part
  (b) was wrong in the opposite direction from predicted, and chasing down
  *why* surfaced a real metric artifact; see Sections 8-9.
- **Statistical check (added 2026-10-21):** Significant at 30 trials; see Section 12.
- **Code:** `experiments/rq5d_misspecification.py` · **Figure:**
  `figures/rq5d_misspecification.png`
- **Builds directly on:** `research_log/RQ3_entropy_vs_random.md` (the
  uncertainty-vs-random comparison reused here) and README's own "Known
  gaps" note: *"Choosing K: we assume K is known; a wrong K is a form of
  misspecification (RQ5)."*

## Plain-English summary (read this first; Sections 1–6 below are the technical record)

Every other RQ5 angle was about *which points to query*: the model could,
in principle, eventually learn the right structure. Misspecification is
different: if the true data has 3 real clusters but the model is only ever
given $K=2$, no amount of querying can ever make it correctly represent all
three. Two of the three true clusters are structurally forced to merge into
one fitted component.

Worked out, not asserted: the naive way to grade this (collapse
the two merged true clusters into one target before scoring) hides the cost
entirely: once the model correctly identifies the merge, everything
scores 100%, correct. The honest metric keeps all three true labels
separate: each fitted component gets a majority-vote identity from the real
points inside it, and any point whose own true label doesn't match that
identity is wrong, permanently, no matter how good the fit becomes. Worked
through by hand before running anything: with 300 points each from three
true clusters (900 total), one cleanly separated and the other two merged
and equal-sized, the best possible permanent ceiling is $300 + 300 = 600$
out of $900$, **66.7%**, not 100%, and not recoverable by any strategy.

**Part (a) confirmed almost exactly: achieved accuracy landed at 66.8%,
right against the predicted 66.7%.** Part (b) went the opposite direction
from predicted, and dramatically: uncertainty sampling beat random in
**30/30 trials (100%)**, not the 40-45% expected from RQ3's baseline.
Diagnosed, not just measured: of 900 points, only **8** are
fixably miscategorized (stray points that crossed the Left
boundary by chance); those 8 average **30x higher entropy** than the other
892, and 7 of the 8 sit in the overall top 20 by entropy. Uncertainty
sampling finds and fixes nearly all of them within its first several
queries; random sampling, with roughly a 0.9% chance per query of hitting
one of the 8, essentially never manages that within 60 queries. Unlike
RQ3's redundant, constantly-regenerating ambiguous region, the useful
information here is a small, *finite* needle in a large haystack, so
targeting wins decisively instead of losing.

**A second, deeper discovery followed from asking why individual runs
exceeded the 66.7% ceiling at all** (uncertainty's own mean: 67.4%, every
single trial above the ceiling, not scattered noise around it). Traced: our "remaining accuracy" convention (grade only still-unlabeled
points, same rule since RQ3) quietly breaks down once some points are
*structurally* unfixable. One trial's full 60 queries broke down as 32
Left, 3 Middle, 25 Right: component 0's identity was locked in as
"Middle," so every one of those 25 Right points, once clamped, is removed
from the grading pool despite being permanently unfixable, not corrected.
Removing a point that can never be right mechanically inflates the
accuracy of whatever remains. Random sampling doesn't get this boost,
since it removes a roughly proportional mix of all three true labels as it
goes. So part of the 100% win rate is the genuine mechanism above, and part
of it is this metric artifact, specific to misspecified settings where
"already asked about" and "correctly resolved" are no longer the same
thing. Full account in Sections 8-9.

## 1. Research question

(Scoped sub-angle of the locked RQ5.) When the model is misspecified
($K=2$ fit to 3 true clusters, two of which are forced to merge): (a) what
is the actual achieved accuracy ceiling, and does it match the predicted
66.7%; (b) within that ceiling, does the uncertainty-vs-random comparison
on the one boundary that's still live (Left vs. the merged bucket) behave
like RQ3's ordinary baseline, or does misspecification change it?

## 2. Hypothesis

**Committed on 2026-10-17, before any code was written:**

> (a) Achieved accuracy will approach the predicted ceiling of 66.7%
> (sanity check on the mechanism, not a strategy comparison).
> (b) Uncertainty sampling's win rate against random sampling will be
> 40-45%, matching RQ3's 43.3% baseline. **Wrong if it falls below 40% or
> above 45%.**

My reasoning at the time, condensed: the only live, queryable
ambiguity left is the ordinary boundary between Left and the merged
bucket: the Middle-vs-Right distinction is invisible to entropy
regardless of strategy, so the comparison should reduce to the same
redundancy mechanics RQ3 already found.

## 3. Experimental setup

$N=900$: 300 points each from three true clusters.

- **Middle and Right:** generated exactly as RQ1-RQ5's two-cluster setup
  ($c=1.0$, $\sigma=1$, means at $(\pm0.5, 0)$), the same overlap used all
  along.
- **Left:** a third cluster at $x=-5$, $\sigma=1$, separation of
  $\approx4.5$ from the nearer of Middle/Right, comfortably past RQ1/RQ2's
  own $c=4$ "well separated, negligible overlap" reference point.
- Model fit with $K=2$ throughout (the misspecification: 2 components for
  3 true clusters).
- **Metric:** majority-vote-per-component accuracy against the full 3-way
  true labels (Section 1 of the plain-English summary), not a collapsed
  many-to-one target. The mapping from true label to fitted component is
  fixed once, right after the initial fit (same stability principle as
  `perm` in RQ3-RQ5c): whichever component's majority identity is "Left"
  receives Left points when clamped; the other component receives *both*
  Middle and Right points when clamped, regardless of which one a point
  truly is; there is no dedicated place for that distinction to go.
- Same budget $T=60$, same warm-starting, same 30 trials, same 5
  random-sampling sub-repeats, as RQ3.

## 4. Independent variable(s)

Strategy $\in\{$uncertainty, random$\}$. $K=2$ (misspecified relative to
the true 3 clusters), cluster sizes and separations held fixed (not swept).

## 5. Dependent variable(s) / metric

Majority-vote-per-component accuracy (Section 3) on the still-unlabeled points,
compared at the final budget for the win/loss count, same role as every
other RQ. Also reported: final achieved accuracy itself, compared
against the hand-derived 66.7% ceiling, as a check on whether the
mechanism understanding is right, not just the strategy
comparison.

## 6. Controls

- **Common random numbers:** both strategies run on the same 30 datasets,
  same initial fit.
- $K=2$, `reg_covar`, EM's `tol` held fixed, matching RQ1-RQ5c.
- Random sampling still needs 5 sub-repeats; uncertainty remains
  deterministic.

## 7. Results

Full output: `experiments/rq5d_misspecification.py`; figure:
`figures/rq5d_misspecification.png`.

| Quantity | Value |
|---|---|
| Predicted ceiling | 66.7% |
| Mean achieved accuracy (pooled) | 66.8% |
| Uncertainty's own mean final accuracy | 67.4% |
| Random's own mean final accuracy | 66.1% |
| Uncertainty beats random | 30/30 = 100.0% |
| Predicted | 40-45% |
| Pre-committed "wrong" line | below 40% or above 45% |
| **Verdict** | **Part (a) held; part (b) wrong (100% vs. 40-45%)** |

Mechanism check (one representative trial): of 900 points, 8 were
genuinely stray (mean entropy 0.458 vs. 0.015 for the rest; 7 of 8 in the
overall top 20 by entropy). Of that same trial's 60 queries, uncertainty
asked about 32 Left, 3 Middle, 25 Right, confirming it spent most of its
budget on the Left-boundary (both fixing genuine strays and continuing to
probe it) plus a large, disproportionate share removing permanently-wrong
Right points from the grading pool.

## 8. Unexpected results

Two, stacked on top of each other. **First:** the complete reversal from
every other RQ5 angle: uncertainty didn't just do better than predicted,
it won *every single trial*, the opposite extreme from the 40-45%
hypothesis. **Second, found only by asking why individual runs exceeded
the theoretical ceiling at all:** the "remaining accuracy" metric
convention, unchanged since RQ3 and never a problem before, has a real
blind spot once some points are structurally unfixable rather than merely
hard. Querying a point and removing it from the grading pool is not the
same thing as correcting it, but our metric has been treating them as
equivalent the whole time, because in every previous RQ, every point
*could* be corrected, so the distinction never mattered.

## 9. Interpretation

**The genuine mechanism (not an artifact):** the useful, fixable
information here (8 points out of 900) is a small, finite set that
doesn't regenerate once resolved, unlike RQ3's large, constantly-refreshing
ambiguous region. Entropy-based targeting is exactly the right tool for
finding a rare, finite needle in a large haystack; blind random guessing
has a very low chance of finding it within a limited budget. This is an
important completion of the RQ3/RQ4/RQ5(a-c) story: uncertainty
sampling isn't universally bad at using its budget; it's bad specifically
when the useful information is diffuse and self-replenishing, and good
when it's sparse and finite. Which case you're in depends on the shape of
the problem, not on uncertainty sampling itself.

**The metric artifact (real, but separate):** once points can be
*permanently* wrong (not just currently wrong), "accuracy measured only on
what's left" stops being a clean, comparable number across strategies that
query different points. A strategy that happens to remove more
permanently-wrong points from the test, without fixing anything, scores
higher, not because it taught the model more, but because it tested
itself on an easier, strategy-selected subset. Uncertainty sampling does
this more than random here, specifically because its targeting rule
doesn't distinguish "this point is uncertain because it's
fixable" from "this point is uncertain because it's stuck in a
permanently-ambiguous bucket." Both look identical to entropy.

**Taken together:** the 100% win rate is real but not a clean read of "how
much better uncertainty sampling teaches the model" the way RQ3's 43.3%
was: it's a mix of fast needle-finding and a scoring convention
that rewards removing unfixable points from the test. A corrected version
of this experiment (Section 11) would need to separate those two effects before
the win-rate number alone could be trusted the way earlier RQs' numbers
were.

## 10. Limitations

- The metric artifact (Sections 8-9) means this RQ's win-rate number is **not**
  comparable to RQ1-RQ5(c)'s win rates in the same simple way;
  it's inflated by an effect those RQs didn't have to contend with. The
  ceiling-confirmation result (part a, 66.8% vs. 66.7%) is unaffected by
  this and stands as a clean result on its own.
- One fixed cluster configuration (sizes, separations), not a sweep. Says
  nothing about partial misspecification (e.g. $K=3$ fit to 4 true
  clusters) or whether the 66.7% ceiling generalizes to unequal cluster
  sizes.
- **Checked while auditing RQ6's tie-breaking bug (2026-10-20):**
  `majority_identity` uses `np.argmax(np.bincount(...))`, which can tie
  exactly on discrete counts the same way RQ6's disagreement score did.
  Verified: 1 exact tie out of 60 component-identity decisions
  across all 30 trials (trial 22, counts `[4, 298, 298]`). Confirmed this
  does not bias the reported results: unlike RQ6, where the tie-break
  decided *which point got queried* (a strategy-dependent choice),
  `identity` here is computed once and shared identically by both
  compared strategies within a trial, and accuracy is symmetric to which
  label wins a tie. The same code pattern recurring here, structurally
  incapable of contaminating this RQ's conclusions.
- The "needle in a haystack" mechanism (Section 9) was confirmed on one
  representative trial's query breakdown, not formally measured across all
  30.

## 11. Next question

**Primary, motivated by Sections 8-9:** a corrected metric that grades
*all* points, every step, including already-clamped ones (so removing a
point from future querying doesn't also remove it from scoring) would
cleanly separate the genuine mechanism from the metric artifact. Worth
re-running with that fix before treating the 100% figure as the real
answer to "does misspecification change the strategy comparison."

**Secondary:** the "small, finite, non-regenerating needle" vs. "large,
self-replenishing haystack" distinction from Section 9 is itself a general,
reusable idea, worth stating as a standalone principle for the eventual
technical report, since it's the cleanest unifying explanation found across
all of RQ3 through RQ5d.

## 12. Statistical check (added 2026-10-21)

Checked after the fact with an exact two-sided binomial test: 30/30 gives p < 0.001 (95% interval for the win rate 0.88 to 1.00), so part (b) is statistically clear and wasn't part of the 100-trial replication. Significance doesn't remove the caveat from Sections 8 and 9, though: part of the 100% comes from the metric artifact, so it still isn't a clean, apples-to-apples win rate.
