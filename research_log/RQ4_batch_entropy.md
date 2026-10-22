# RQ4: Does top-k entropy produce an informative batch, or a redundant one?

- **Date started / run:** 2026-10-14
- **Status:** done. Hypothesis held, and the effect was a bit stronger than
  predicted; see Section 8.
- **Replication (added 2026-10-21):** Inconclusive at 100 new trials (56/100 with 7 ties, p = 0.06); see Section 12.
- **Code:** `experiments/rq4_batch_entropy.py` · **Figure:**
  `figures/rq4_batch_entropy.png`
- **Builds directly on:** `research_log/RQ3_entropy_vs_random.md` Section 9's
  redundancy finding (a thin, ~55-point ambiguous neighborhood consumed most
  of a 60-query budget one-at-a-time) and `lessons/04_clamped_em_and_active_learning.md`.

## Plain-English summary (read this first; Sections 1–6 below are the technical record)

RQ3 found that querying one point at a time, always the most uncertain,
keeps circling back to the same thin, similar-looking neighborhood, and each
extra point from that neighborhood mostly confirms what the first one
already taught, rather than adding something new. RQ4 asks the natural
next question: if that's true for *sequential* querying, what happens if we
grab the top-$k$ most uncertain points **all at once**, before any of them
have been revealed? Do they turn out to be spread out and varied, or does
the batch just hand us 10 near-duplicates of the same redundant information?

**The hypothesis held, and a bit more strongly than predicted.** Sequential
beat batch in 20 of 30 trials (66.7%), above the predicted 50-60% range,
not just inside it, and well clear of the 40% "wrong" line. Batching the
same entropy ranking into one stale snapshot does cost accuracy
compared to refreshing the ranking after every single reveal.

## 1. Research question

(Locked wording, README.) Does top-$k$ entropy produce an informative batch,
or a redundant one?

Concretely: compare the same total query budget spent two ways: one point
at a time with a refit in between each (RQ3's "uncertainty sampling,"
already run), versus the top-$k$ highest-entropy points revealed all at once
per round, refitting only once per round (new). Same selection criterion
(entropy) in both; the only thing that changes is whether the ranking gets
refreshed between individual reveals or stays stale across a whole batch.
This deliberately does **not** re-test entropy-vs-random: RQ3 already
answered that question; RQ4 isolates the cost of batching itself.

## 2. Hypothesis

**Committed on 2026-10-14, before any code was written:**

> Sequential (one-at-a-time, refit between each) beats batch (top-10-at-once,
> refit once per round of 10) in 50-60% of 30 independent dataset repeats, at
> the same total budget (60 queries). **Wrong if sequential wins in fewer
> than 40% of repeats.**

My reasoning behind the prediction, condensed: a
batch chosen from a single, stale entropy ranking will mostly grab points
from the same narrow ambiguous neighborhood RQ3 already found, where each
additional point in that batch "will mostly confirm what you already
learned from the first, adding nothing new." Sequential avoids this because
it refreshes the ranking after every single reveal, so it can't pick the
same near-duplicate twice.

## 3. Experimental setup

Same $c=1.0$, $N=600$, initial `fit_best_of` ($R=10$, $n\_iter=2000$) as RQ3,
and, deliberately, **the same 30 trial seeds as RQ3** (`data_seed =
trial_idx`, `restart_seed = trial_idx + 10_000`), so the sequential condition
reproduces RQ3's "uncertainty sampling" curve exactly (common random
numbers, one level up: not just the same data, the same *function call*,
reused directly from `experiments/rq3_entropy_vs_random.py` rather than
re-implemented, so the two conditions can't silently drift apart from two
independently written loops).

- **Sequential condition:** `rq3.run_query_loop(..., "uncertainty", None)`,
  unchanged, budget $T=60$.
- **Batch condition (new):** for 6 rounds of $k=10$: rank all still-unlabeled
  points by entropy under the *current* parameters, reveal and clamp the
  top 10 simultaneously, warm-started refit once ($n\_iter=200$), record
  held-out accuracy. $6\times10=60$, matching the sequential budget exactly.
- Both conditions are **deterministic** given the shared initial fit (no
  randomness in either selection rule); unlike RQ3's random-sampling arm,
  neither needs sub-repeats.

## 4. Independent variable(s)

Condition $\in\{$sequential, batch$\}$. $c=1.0$, total budget $T=60$,
$k=10$ held fixed (not swept here).

## 5. Dependent variable(s) / metric

Same as RQ3, for the same reasons already established there: **accuracy on
the still-unlabeled points only**, compared at the final budget (60 queries)
for the win/loss count, and plotted across the budget for the figure
(sequential: all 60 steps; batch: the 6 round-checkpoints at 10, 20, ..., 60).

## 6. Controls

- **Common random numbers:** the same 30 dataset/restart seeds as RQ3,
  reused directly, isolating "batched vs. sequential" as the only
  difference, not also comparing against a different set of datasets.
- **Same selection criterion in both conditions** (entropy), isolating the
  batching mechanism itself, not entropy-vs-random (already answered, RQ3).
- $N=600$, $K=2$, $\sigma=1$, `reg_covar`, EM's `tol` held fixed, matching
  RQ1/RQ2/RQ3.

## 7. Results

Full output: `experiments/rq4_batch_entropy.py`; figure:
`figures/rq4_batch_entropy.png`.

| Quantity | Value |
|---|---|
| Sequential beats batch | 20/30 = 66.7% |
| Predicted | 50-60% |
| Pre-committed "wrong" line | below 40% |
| **Verdict** | **Hypothesis held** (66.7% > 40%, and above the predicted range) |

Sanity check before trusting this: trial 0's sequential final accuracy
(0.6778) matched RQ3's own trial-0 diagnostic from the earlier conversation
exactly, confirming the reused `rq3.run_query_loop` call reproduces RQ3's
curve byte-for-byte rather than silently drifting from it.

## 8. Unexpected results

**Not just confirmed: a bit stronger than predicted.** 66.7% sits above the
entire 50-60% range committed in advance, not merely inside it. The
redundancy cost of batching a stale entropy ranking is real and, if
anything, somewhat larger than the reasoning behind the hypothesis
suggested.

One implementation issue worth recording honestly, not as a finding but as
a process note: the first full run of this script failed immediately with
`ModuleNotFoundError: No module named 'experiments'`. Cause: running
`python experiments/rq4_batch_entropy.py` directly puts the script's own
folder on the import path, not the project root, so
`import experiments.rq3_entropy_vs_random` couldn't find its own package.
Fixed with a small `sys.path` bootstrap at the top of the file before that
import; re-verified with a direct invocation (not just `python -c`, which
had masked the problem in the smoke test) before re-running for real.

## 9. Interpretation

Same mechanism as RQ3, Section 9, now confirmed from the other direction. Sequential
refreshes its entropy ranking after every single reveal, so it can never
pick the same near-duplicate point twice; each reveal shifts the model
just enough that the *next* most-uncertain point is usually somewhere
slightly different. Batch commits to all 10 picks from one stale snapshot,
before any of them have taught the model anything, so if several of those
10 points are sitting in the same narrow neighborhood (as RQ3's count
suggested they would be), the batch spends several of its 10 "questions"
re-confirming the same thing, where sequential would have already moved on
to a different part of the data by its 2nd or 3rd reveal.

## 10. Limitations

- One fixed $k=10$, one fixed $c=1.0$, one fixed total budget ($T=60$); none
  swept. A much smaller $k$ (closer to sequential) or a much larger one
  (more extreme staleness) could change the size of the gap.
- 30 trials, each compared once at the final budget: real effect, but not
  a high-precision estimate of the exact size of the gap.
- This only tests *plain* top-$k$ entropy batching. Says nothing yet about
  whether a diversity-aware batch, e.g. combining entropy with a
  k-means++-style spread rule, so the batch is forced to cover different
  neighborhoods instead of the single most-ambiguous one, would recover
  some or all of this gap.

## 11. Next question

**Primary:** does a diversity-aware batch selection rule close the gap found
here? Concretely: instead of picking the raw top-10 by entropy, pick the
first point by entropy as before, then pick each subsequent point by entropy
*weighted by distance from already-chosen batch members* (the same
rich-gets-richer rule k-means++ uses for spreading out starting means,
Lesson "k-means++" discussion, applied here to batch selection instead of
initialization). If that closes most of the gap, it would mean the problem
is simply "plain top-$k$ doesn't consider diversity," not something deeper
about batching itself.

**Secondary:** RQ5's confidently-wrong blind spot (Lesson 4, Section 4.4) hasn't
been tested in combination with batching yet: does a lopsided initial fit
make the batch redundancy problem worse, better, or about the same, the way
RQ3, Section 8 found it didn't clearly matter for the sequential case?

## 12. Statistical check and replication (added 2026-10-21)

The original 20/30 wasn't significant on its own (exact two-sided binomial test, p = 0.10; 95% interval 0.47 to 0.83). I reran the comparison unchanged, through the experiment's own `run_trial`, on 100 new trials whose seeds don't overlap the originals (`experiments/rq_replication.py`; per-trial results in `data/replication_results.json`). The criterion was written down before the rerun: an effect counts as replicated only if the new win rate lands on the same side of 50% as the original and a two-sided sign test on the new trials (ties dropped) gives p < 0.05.

**Result:** sequential won 56, lost 37, and tied 7 (sign test p = 0.06). The average accuracy gap was 0.7 points in sequential's favour.

**What this changes:** the pre-registered verdict (held) stands on its own terms, but the effect isn't established. Refreshing the ranking after every reveal looks slightly better than batching, by less than a point of accuracy on average, and that difference didn't clear the bar I set.
