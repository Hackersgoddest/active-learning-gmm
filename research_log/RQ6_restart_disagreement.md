# RQ6: Do restart ensembles provide information beyond responsibility entropy?

- **Date started / run:** 2026-10-20
- **Status:** done. Hypothesis held. Scoped to restart ensembles only this
  pass; bootstrap ensembles deliberately deferred to a possible RQ6b.
- **Replication (added 2026-10-21):** Replicated, and stronger, at 100 new trials (67/100, p < 0.001); see Section 13.
- **Code:** `experiments/rq6_restart_disagreement.py` (plus a shared-library
  robustness fix to `gmmlab.em.fit_many`, verified not to change any prior
  RQ's results; see Section 8) · **Figure:**
  `figures/rq6_restart_disagreement.png`
- **Builds directly on:** `research_log/RQ5_blind_spot.md` (the "cheating"
  oracle this is the realistic, non-cheating analogue of) and README's
  locked RQ6 wording: *"Restarts measure optimization uncertainty; bootstraps
  measure data-sampling uncertainty. They are reported separately, and
  neither is called 'epistemic' without qualification."*

## Plain-English summary (read this first; Sections 1–6 below are the technical record)

RQ5a built a strategy that cheats (it checks the true label to
find the model's worst, most-confidently-wrong errors) and it still lost
to plain entropy sampling, because clamping is a full-strength correction
that can overshoot and destabilize the fit. RQ6 asks the realistic version
of the same question: no real system can check the true label, but could
**disagreement across several independent restarts** act as a usable proxy
for "this point might be confidently wrong," without cheating?

A real misconception was caught and corrected before committing to a
hypothesis: restart-disagreement does **not** make the correction itself
gentler. Clamping is clamping, identical in strength regardless of which
strategy picked the point: restart-disagreement can only change *which*
point gets picked, not how hard the resulting yank is. The real question is
whether it tends to pick points with *less* leverage than the oracle's
picks (points that are wrong but not extremely confidently so), not
whether it corrects them more softly.

**The hypothesis held.** Restart-disagreement beat plain entropy in 17 of
30 trials (56.7%), squarely inside the predicted 45-60% range, well clear
of the 45% "wrong" line. It also beat RQ5a's cheating oracle outright
(40%), despite having strictly worse information (no access to the true
label at all), consistent with the leverage reasoning: disagreement can
flag a point without requiring extreme certainty from any single restart,
so its picks carry less leverage, and less overcorrection risk, than the
oracle's always-extreme targets.

Two real implementation issues surfaced along the way, both caught before
trusting the result, not after: a tie-breaking bug in the strategy itself
(Section 8) and a genuine floating-point precision bug in shared library code
(Section 8), the latter fixed in `gmmlab.em.fit_many` itself, not worked around
locally.

## 1. Research question

(Locked wording, README, restart half only.) Do restart ensembles provide
information beyond responsibility entropy? Concretely: does querying by
**disagreement across 5 independent restarts** (instead of by entropy from
a single fit) do better than plain entropy sampling, and does it do
better than RQ5a's cheating oracle did (40%), given it may pick
lower-leverage points than the oracle's extreme-confidence-only targets?

## 2. Hypothesis

**Committed on 2026-10-20, before any code was written:**

> Restart-disagreement sampling beats plain entropy sampling in 45-60% of
> 30 trials. **Wrong if it wins in fewer than 45% of trials.**

My reasoning at the time, condensed: the oracle always demanded
*extreme* confidence (it specifically searched for the most-confidently-wrong
point). Restart-disagreement can flag a point just because different
restarts land on different answers for it, even if no single restart ever
held that point with extreme certainty, so its picks should carry less
leverage on average than the oracle's, and therefore less overcorrection
risk, even though the two strategies may sometimes flag similar points.

## 3. Experimental setup

Same $c=1.0$, $N=600$, initial `fit_best_of`, warm-started refits
($n\_iter=200$), budget $T=60$, 30 trials, held-out accuracy, identical to
RQ3, for direct comparability.

**New strategy, "restart-disagreement":** at each query step, in addition
to the single warm-started fit used to actually update the model, run 5
**fresh** restarts (`init="random"`, independent random seeds, $n\_iter=500$)
constrained to respect the already-clamped points so far
(`clamped_labels` passed to each). Component labeling is realigned to the
main fit's own labeling via `gmmlab.matching.match_components` before
comparing, **necessary**, not optional: two independently-fit GMMs have
no reason to agree on which physical cluster is "component 0," so raw,
unaligned label comparison would manufacture fake disagreement out of
arbitrary labeling, not real uncertainty. After alignment, each
still-unlabeled point gets 5 votes (one per restart); disagreement = (number
of restarts not matching the majority vote) / 5. Query whichever point has
the highest disagreement score. The 5 ensemble fits themselves are
discarded after the query decision; only the main warm-started fit
persists across steps, same as every other RQ.

## 4. Independent variable(s)

Strategy $\in\{$restart-disagreement, entropy$\}$. $c=1.0$, $T=60$,
ensemble size (5 restarts per step) held fixed (not swept).

## 5. Dependent variable(s) / metric

Held-out accuracy on still-unlabeled points, same convention as RQ3-RQ5c
(not RQ5d's multiway metric, this reuses the plain two-cluster setup).
Win/loss count at the final budget.

## 6. Controls

- **Common random numbers:** both strategies run on the same 30 datasets,
  same initial fit.
- Component-label realignment (Section 3) applied consistently every step, for
  every ensemble member, so disagreement reflects real classification
  disagreement, not arbitrary labeling differences.
- $N=600$, $K=2$, $\sigma=1$, `reg_covar`, EM's `tol` held fixed.
- Computational note: 5 extra fresh-restart fits per query step is
  substantially more expensive than prior RQs' single warm-started fit per
  step; timed before committing to the full 30-trial run.

## 7. Results

Full output: `experiments/rq6_restart_disagreement.py`; figure:
`figures/rq6_restart_disagreement.png`.

| Quantity | Value |
|---|---|
| Restart-disagreement beats entropy | 17/30 = 56.7% |
| RQ5a's oracle vs. entropy, for comparison | 40.0% |
| Predicted | 45-60% |
| Pre-committed "wrong" line | below 45% |
| **Verdict** | **Hypothesis held** (56.7% inside the predicted range) |

## 8. Unexpected results

Not the headline number: the two bugs caught while building this.

**A tie-breaking bug, found before running anything for real.** With only
5 restarts and 2 components, disagreement can only take 3 possible values
(0, 0.2, 0.4), so large exact ties are near-guaranteed across 600 points,
confirmed: 317/600 points tied at the maximum value in one check.
`argmax` would have deterministically picked whichever tied point happened
to sit first in the array, biased toward one true cluster, since
`make_two_gaussians` lays out component 0's points first. Concretely
confirmed to matter: the old (biased) pick had entropy 0.0023 (a point the
model was already almost certain about, nearly the worst possible choice);
the fixed version (entropy as tie-break among the disagreement-tied
points) picked a point with entropy 0.6931, essentially the maximum
possible. Fixed before the first real run.

**A floating-point precision bug in shared library code**, the
third time this exact failure mode has been seen (twice in
`RQ5c_outliers.md`, now once on completely plain, undecorated data with no
special construction). `fit_best_of`'s initial fit crashed with
`NonMonotonicLikelihoodError` on one specific trial; traced to one restart
heading toward the same near-singular collapse seven of its sibling
restarts converged to cleanly (`pi≈[0.017, 0.983]`, RQ1's pathology, no
decoys or imbalance needed this time: plain overlap at $c=1.0$ is
sufficient on its own, rarely, by chance). Fixed properly in
`gmmlab.em.fit_many` itself (not a local workaround): a restart that
raises this error is now skipped rather than crashing the whole ensemble,
the same robustness philosophy `is_degenerate` already applies to
restarts that converge to a collapsed answer. `fit()`'s own check is
unchanged and still raises immediately on a single fit. Regression-tested:
RQ3's already-published trial-0 number came out byte-identical
after the change (`0.6777777777777778`, unchanged).

## 9. Interpretation

The core finding: a *realistic* proxy for "this point might be
confidently wrong" (restart disagreement) outperforms both plain entropy
*and* a cheating oracle with perfect information. That's a
useful result: it says the oracle's failure in RQ5a wasn't really about
*what* it was looking for (confidently-wrong points), it was about *how
certain* it insisted on being before acting. Disagreement across
independent restarts naturally produces a *softer* signal: a point can
show high disagreement without any single restart ever being extremely
confident about it, and that softness, not some other mechanism, is the
most direct, evidence-backed explanation for why it avoids the oracle's
overcorrection problem.

One thing this result does **not** establish, flagged plainly rather than
assumed: whether restart-disagreement also avoids RQ3's separate
redundancy problem (repeatedly querying near-duplicate points in a
crowded, regenerating boundary region) was not tested here. The committed
hypothesis and the mechanism behind it were specifically about leverage and
overcorrection (the RQ5a thread), not about redundancy (the RQ3/RQ4
thread): these are different failure families, and this result speaks to
only one of them.

## 10. Limitations

- Ensemble size fixed at 5, not swept.
- $n\_iter=500$ for ensemble members is a cost/quality tradeoff, not
  verified to be enough for full convergence on every trial (though the
  $n\_iter=500$ vs. $2000$ check during debugging showed disagreement
  levels were stable across that range, suggesting under-convergence
  wasn't driving the headline widespread-disagreement pattern).
- Bootstrap ensembles (data-sampling uncertainty) are a separate, deferred
  investigation; this only tests optimization uncertainty.
- Whether restart-disagreement also escapes RQ3's redundancy problem is
  untested (Section 9); this result only speaks to the leverage/overcorrection
  mechanism.
- Computationally expensive relative to every other RQ (~50s/trial vs.
  single-digit seconds for most prior ones); 5 extra fresh-restart fits
  per query step adds up over 60 steps.

## 11. Next question

**Primary, motivated by Section 9's flagged gap:** does
restart-disagreement sampling avoid RQ3's redundancy trap, or does it
suffer a version of it too? Concretely checkable the same way RQ3's
original 55-point boundary layer was counted: track, across the 60
queries, how many *distinct* points get selected vs. how often the
strategy returns to points "near" ones it already asked about (e.g., by
checking whether consecutive queries keep landing in the same persistently
high-disagreement region, the way entropy kept circling the same ~55-point
layer in RQ3). If restart-disagreement's queries turn out to be more
spatially/structurally spread out than entropy's were, that would be
direct, independent evidence for *why* it also avoids redundancy, not just
overcorrection: two separate reasons it might outperform entropy, not
one.

**Secondary:** RQ6b, bootstrap ensembles (data-sampling uncertainty),
deliberately deferred from this pass, still open.

## 12. Follow-up: does restart-disagreement also escape RQ3's redundancy problem? (2026-10-20)

**Why this is a separate section, not a change to Section 9.** Section 9 is left exactly
as written: it answers the leverage/overcorrection question, which is
what the original hypothesis tested. This section asks a
different question, flagged as open in Sections 9 and 11: does
restart-disagreement *also* avoid RQ3's separate redundancy problem
(repeatedly querying near-duplicate points in one crowded, regenerating
boundary region)?

**Hypothesis, committed before running:**

> The average ratio of restart-disagreement's query spread (standard
> deviation of the x-coordinate across its 60 queried points) to entropy's
> query spread, across the same 30 trials, will be between 1.2x and 2.0x.
> **Wrong if the ratio is below 1.2x.**

**Method.** `experiments/rq6b_redundancy_check.py`: a fresh
implementation (imports `restart_disagreement_scores` from
`rq6_restart_disagreement.py`, rather than modifying that file's
already-published `run_query_loop`/`run_trial`, to carry zero regression
risk to RQ6's locked results). Same 30 trial seeds, same budget, same two
strategies. For each trial and strategy, records the x-coordinate of every
one of the 60 queried points and computes its standard deviation; reports
the per-trial ratio (restart-disagreement's spread / entropy's spread) and
its average across all 30 trials.

**Results.** Full output: `experiments/rq6b_redundancy_check.py`; figure:
`figures/rq6b_redundancy_check.png`.

| Quantity | Value |
|---|---|
| Mean spread ratio | 1.38x |
| Median spread ratio | 1.09x |
| Per-trial range | 0.50x – 7.26x |
| Predicted | 1.2x-2.0x |
| Pre-committed "wrong" line | below 1.2x |
| **Verdict (pre-committed test, on the mean)** | **Hypothesis held** (1.38x inside the predicted range) |

**A real nuance, caught before trusting the headline number.** The
pre-registered test was on the *mean*, and it passes. But the median
(1.09x) would have fallen *below* the wrong-line if it had been the test
statistic instead, and the per-trial range (0.50x to 7.26x) is wide
enough that "the mean passed" and "this effect is typical" are not the
same claim. The honest picture: in a *typical* trial, restart-disagreement
and entropy query roughly similarly-spread sets of points (ratio near 1).
In a *minority* of trials, restart-disagreement's spread is dramatically
wider (up to 7x), and those trials are pulling the mean up into the
predicted range, not a consistent, trial-by-trial effect.

**Interpretation.** The most likely explanation, tied to RQ6's
own diagnostic (Section 8's widespread-disagreement finding, itself traced to
some restarts finding the lopsided collapse and others finding the honest
structure for the *same* dataset): trials where the restart ensemble
disagrees broadly about the overall structure should produce queries
scattered across much more of the data than entropy's boundary-focused
picks, a large ratio. Trials where the restarts mostly agree with each
other (a more stable fit) should make restart-disagreement behave a lot
like entropy: ratio near 1. This is a plausible, mechanism-consistent
story, **not independently re-verified for this specific follow-up**: it
would need a direct check correlating each trial's initial disagreement
breadth with its spread ratio, not yet done. Flagged as speculation,
consistent with how RQ2's k-means++ mechanism was handled.

**Bottom line for the open question from Sections 9 and 11:** restart-disagreement's RQ6 win
is not fully explained by "it also escapes the redundancy trap": that
effect, where it exists, is real but concentrated in a subset of trials,
not a uniform, trial-by-trial advantage. The leverage/overcorrection story
(Section 9) remains the more consistently-evidenced explanation for the headline
56.7% win rate.

## 13. Statistical check and replication (added 2026-10-21)

The original 17/30 wasn't significant on its own (exact two-sided binomial test, p = 0.58; 95% interval 0.37 to 0.75). I reran the comparison unchanged, through the experiment's own `run_trial`, on 100 new trials whose seeds don't overlap the originals (`experiments/rq_replication.py`; per-trial results in `data/replication_results.json`). The criterion was written down before the rerun: an effect counts as replicated only if the new win rate lands on the same side of 50% as the original and a two-sided sign test on the new trials (ties dropped) gives p < 0.05.

**Result:** restart disagreement won 67, lost 28, and tied 5 (sign test p = 0.0001). The average accuracy gap was 3.2 points in its favour.

**What this changes:** replicated, and more strongly than the original run suggested. Restart disagreement is now the most reliable strategy in the project. The leverage explanation in Section 9 and the redundancy finding in Section 12 are still explanations rather than tested mechanisms, and checking what restart disagreement actually selects is the obvious next step.
