# RQ3: Under what conditions does responsibility entropy improve learning efficiency relative to random selection when labels are limited?

- **Date started / run:** 2026-10-14
- **Status:** done. Hypothesis wrong by its own pre-committed line.
- **Replication (added 2026-10-21):** Inconclusive at 100 new trials (43/100, p = 0.19); see Section 12.
- **Code:** `experiments/rq3_entropy_vs_random.py` · **Figure:**
  `figures/rq3_entropy_vs_random.png`
- **Builds directly on:** `research_log/RQ1_cluster_overlap.md` (the
  confident-but-wrong fit), `research_log/RQ2_initialization.md` (initialization
  sensitivity, and its own open "mechanism" follow-up), and
  `lessons/04_clamped_em_and_active_learning.md` (what clamping a label does to
  the model, and the leverage argument this experiment actually tests).

## Plain-English summary (read this first; Sections 1–11 below are the technical record)

The plan was built piece by piece, by reasoning through what could go wrong
with each naive choice, before any code was written (per house rule 4:
"before RQ3, the analysis plan is written into the log first").

Three wrong-but-tempting metric choices were caught and rejected along the
way: **log-likelihood** (RQ1 already proved it can favor a wrong, lopsided fit
over the honest one: see `em.is_degenerate`'s docstring, $\pi=[0.007,0.993]$
beating the balanced fit); **the RQ1-style "gap"** (redundant here, since
Bayes accuracy is a fixed constant at one fixed $c$, so it would track plain
accuracy exactly, just shifted); and **accuracy over the whole dataset
including already-revealed points** (proven, by pushing to the extreme of
revealing every point, to trivially hit 100% for *both* strategies, telling
us nothing). The metric that survived: **accuracy on the still-unlabeled
points only**, the standard "don't grade yourself on the answers you were
given" principle.

**The headline result: the hypothesis was wrong, and not narrowly.**
Uncertainty sampling beat random sampling in only 13 of 30 trials (43.3%),
below even the pre-committed "I was wrong" line of 50%, nowhere near the
60-70% predicted. Uncertainty sampling, on average, is **worse**
than picking a random point to ask about, in this setting.

**Why, diagnosed concretely, not just asserted:** uncertainty sampling's own
rule (always query whichever point is currently *most* ambiguous) makes it
spend most of a 60-query budget revisiting a thin, similar-looking boundary
neighborhood. A direct count on one trial's initial fit: of 600 points, only
3 sat within $0.001$ of the true maximum entropy, 30 more were "moderately"
divided, 22 "leaning," and **544 were already confidently assigned**, the
bulk of the dataset, including, in a lopsided fit, a large block that's
confidently *wrong*. Uncertainty sampling exhausts the thin ambiguous layer
(roughly 55 points here) long before it would ever reach that 544-point bulk.
Random sampling has no such redundancy problem: by chance, its 60 queries
spread across the whole dataset, occasionally correcting large-scale
structure uncertainty sampling's own rule keeps it from ever reaching.

## 1. Research question

(Locked wording, README.) Under what conditions does responsibility entropy
improve learning efficiency relative to random selection when labels are
limited?

Concretely: starting from a GMM fit on mostly-unlabeled, overlapping data,
and allowed to reveal a limited number of points' true labels one at a time
(clamping each, per `lessons/04`), does always revealing the
highest-entropy point ("uncertainty sampling") produce a more accurate model,
after the same number of questions, than revealing a random point ("random
sampling")?

## 2. Hypothesis

**Committed on 2026-10-14, before the code was run, and written
into this file afterward (a documentation lag, not a re-guess: the numbers
below are exactly what I committed to before `experiments/rq3_entropy_vs_random.py`
was ever executed).**

> Uncertainty sampling achieves higher held-out accuracy than random sampling
> in 60-70% of 30 independent dataset repeats (query budget $T=60$, $c=1.0$).
> **Wrong if it wins in fewer than 50% of repeats.**

(An earlier draft of this hypothesis was internally inconsistent: it predicted
"60-70%" but set the wrong-line at "random wins more than 30% of the time,"
which is the same as "uncertainty wins less than 70%", meaning the predicted
range mostly overlapped with the "I was wrong" zone. Caught and fixed before
running: the wrong-line was lowered to 50%, leaving the 60-70% prediction
sitting clearly above it.)

## 3. Experimental setup

$c=1.0$, $N=600$ (300/component), query budget $T=60$ (10% of $N$), warm-started
EM throughout (justified below). The query loop, in order:

1. Generate one dataset (`gmmlab.datasets.make_two_gaussians`, same generator
   as RQ1/RQ2) at $c=1.0$, RQ1/RQ2's "common, not universal" overlap case,
   enough uncertainty *and* enough of the confident-but-wrong pattern
   for the comparison to mean something; $c=4$ would give nothing interesting
   to query.
2. Fit an initial GMM (`gmmlab.em.fit_best_of`, $R=10$, $n\_iter=2000$,
   matching RQ1/RQ2), all points unlabeled.
3. Repeat for $T=60$ queries:
   a. **Strategy branch.** Uncertainty sampling: pick the still-unlabeled point
      with the highest entropy $H_n$ (Lesson 4, Section 4.3). Random sampling: pick a
      uniformly random still-unlabeled point.
   b. Reveal that point's true label, clamp its responsibility (Lesson 4,
      Section 4.1).
   c. **Warm-start, don't re-initialize**: refit from the *current* parameters
      with the new clamp added ($n\_iter=200$), rather than restarting from
      fresh random restarts. Decided before running: refitting from scratch
      every query would re-randomize initialization at every step, risking EM
      landing in a different local optimum purely from re-init luck, a
      confound unrelated to which strategy picked the point. Warm-starting
      also matches Lesson 4's picture (a label nudges the *existing*
      model). `gmmlab.em.fit` gained `init_params` (skip the init strategy,
      start from given parameters) and `clamped_labels` (fix a point's
      responsibility to a one-hot vector, every E-step, per Lesson 4, Section 4.1) to
      support this.
   d. Record accuracy on the **remaining unlabeled points only** (Section 5).
4. Produces one accuracy-vs-queries-asked curve per strategy, per trial.

**A theoretical wrinkle, decided before running:** Lesson 2, Section 2.6's proof that
log-likelihood never decreases was derived for the standard, fully-unlabeled
E-step. It does not automatically extend once some responsibilities are fixed
rather than recomputed by Bayes' rule: forcing a point against the model's
current belief can trade off unconstrained fit quality for correctness on
that one point, so a real drop is possible under clamping and must not be
treated as a bug. `NonMonotonicLikelihoodError` is accordingly only raised
when `clamped_labels` is `None`; every ordinary, fully-unlabeled fit (all of
RQ1/RQ2) keeps the exact same strict check as before.

## 4. Independent variable(s)

Strategy $\in\{$uncertainty, random$\}$. $c=1.0$ and $T=60$ held fixed (not
swept here; RQ1/RQ2 already swept $c$).

## 5. Dependent variable(s) / metric

**Accuracy on the still-unlabeled points only**, recomputed after each query,
for each strategy. Three alternatives were considered and rejected first,
each for a concrete, demonstrated reason:

- *Log-likelihood*: already shown in RQ1 to prefer a wrong, lopsided fit over
  an honest one ($\pi=[0.007,0.993]$ beat the balanced fit). Using it here
  risks rewarding a strategy for steering toward exactly that pathology, not
  for learning the true structure.
- *RQ1's "gap" (EM accuracy $-$ Bayes accuracy)*: Bayes accuracy depends only
  on $c$, which is fixed throughout this experiment, so the gap and plain
  accuracy differ only by a constant and would rank the two strategies
  identically. No reason to carry the extra term here (it mattered in RQ1
  because $c$ varied there).
- *Accuracy over all points, including clamped ones*: pushed to the extreme
  (reveal every point) to show it fails: both strategies trivially reach
  100% once everything is labeled, which proves the metric is contaminated by
  "free," guaranteed-correct points well before that extreme too. Rejected in
  favor of held-out accuracy, the standard fix.

## 6. Controls

- **Common random numbers, one level up from RQ1/RQ2:** for a given repeat,
  generate *one* dataset and run **both** strategies on it, starting from the
  same initial fit, isolating the query strategy as the only difference
  between the two runs of that repeat, same logic as RQ1, Section 3 / RQ2, Section 6.
- **Repeats across datasets: 30** independent datasets (matching RQ1/RQ2's
  own count); a single dataset's curve could just be a lucky or unlucky
  draw.
- **Repeats within a dataset, for random sampling specifically: 5.**
  Uncertainty sampling is deterministic given the starting fit (always the
  max-entropy point, no internal randomness to average over, confirmed: its
  log-likelihood history never touches `rng`). Random sampling's choice of
  *which* point to query is itself randomized, so it needs several repeated
  runs even on one fixed dataset to find its typical performance; its 5
  sub-repeats' final accuracies are averaged before comparing to
  uncertainty's single run.
- $N=600$, $K=2$, $\sigma=1$, `reg_covar`, EM's `tol` held fixed, matching
  RQ1/RQ2's values.

## 7. Results

Full output: `experiments/rq3_entropy_vs_random.py`; figure:
`figures/rq3_entropy_vs_random.png`.

| Quantity | Value |
|---|---|
| Uncertainty sampling wins | 13/30 = 43.3% |
| Predicted | 60-70% |
| Pre-committed "wrong" line | below 50% |
| **Verdict** | **Hypothesis wrong** (43.3% < 50%) |
| started lopsided ($\min_k\pi_k<0.10$ at the initial fit) | 7/15 = 46.7% |
| started balanced | 6/15 = 40.0% |

The lopsided/balanced breakdown is a **post-hoc diagnostic, not part of the
pre-registered test**, decided during the run, after a smoke test surfaced
the blind-spot behavior below, to help explain whatever the headline number
turned out to be, the same role RQ1/RQ2's "Interpretation" sections play.
Verified not to be a code asymmetry first: the two strategies' query loops
are identical in every line except which index gets picked next (same
budget, same warm-started refit, same held-out metric); random's 5
sub-repeats on one trial gave different values (0.670, 0.676,
0.657, 0.657, 0.669), confirming no RNG-reuse bug was silently freezing them.

## 8. Unexpected results

**The hypothesis is wrong, and not narrowly: uncertainty sampling
loses to random sampling overall (43.3% vs. an even 50%).** This is the
opposite of the predicted direction, not just a smaller-than-expected win.

**The breakdown held a second surprise.** Before running, the expectation
(from the smoke test, where uncertainty sampling visibly got stuck circling
~55 boundary points in a lopsided trial) was that "started lopsided" would be
*clearly worse* for uncertainty sampling than "started balanced", since the
specific blind-spot mechanism (Lesson 4, Section 4.4) predicts it should matter most
exactly there. Instead, both groups lost to random by similar margins
(46.7% vs. 40.0%, $n=15$ each, a difference well within noise at this
sample size). The extreme collapse pathology is **not** the main story;
whatever is costing uncertainty sampling shows up with or without it.

## 9. Interpretation

The mechanism, worked out from a concrete count on one trial's
initial fit (600 points): only 3 sat within $0.001$ of the true maximum
entropy, 30 more were "moderately" divided, 22 "leaning," and **544 were
already confidently assigned**. Uncertainty sampling's rule (always query
whichever point is currently *most* ambiguous) exhausts that thin layer
(roughly 55 points) long before a 60-query budget would ever reach the
544-point bulk, where the real large-scale structure (and, when present, any
lopsided collapse) lives.

Crucially, this doesn't depend on the *extreme* collapse pathology being
present; it only needs a thin, ambiguous boundary layer to exist
at all, which any two overlapping Gaussians naturally produce. That's why
"started balanced" trials lost almost as often as "started lopsided" ones
(Section 8): the redundancy problem is more general than the specific failure mode
RQ1 found. Lesson 4, Section 4.3's leverage argument (an uncertain point moves the
model *more* per clamp) is still true, but leverage over the parameters is
not the same as *new information*: once the first few reveals have pinned
down roughly where the boundary sits, further reveals from the same
neighborhood are largely redundant. Random sampling has no such redundancy:
by chance, its budget spreads across the whole dataset, occasionally
correcting structure uncertainty sampling's own rule keeps it from reaching.

## 10. Limitations

- One fixed $c=1.0$, one fixed budget $T=60$; neither was swept. A much larger or
  much smaller budget, or a different overlap level, could change the
  picture (e.g. a large enough budget would eventually exhaust the ambiguous
  layer and start touching the bulk regardless of strategy).
- The lopsided/balanced breakdown (Sections 7-8) is $n=15$ per group, a real but
  small sample; the near-tie between groups should not be read as "exactly
  equal," only as "not clearly different at this size."
- This tests one-at-a-time querying with a full refit between each reveal.
  Says nothing yet about batches (RQ4) or about whether uncertainty
  sampling's blind spot specifically costs it when the collapse pathology is
  more severe than tested here (RQ5).

## 11. Next question

**Primary, motivated by Section 9:** RQ4 ("does top-$k$ entropy produce an
informative batch, or a redundant one?") is no longer just the next item on
the roadmap: this result is a one-at-a-time preview of exactly that
redundancy concern. Worth testing: does picking the top-$k$ most
uncertain points *at once* suffer the same "thin neighborhood" problem, or
does batching change the picture?

**Secondary:** RQ5's confidently-wrong blind spot (Lesson 4, Section 4.4) was
expected to be the dominant mechanism here and, per Section 8, does not appear to
be: the redundancy problem is more general. Worth revisiting once RQ5 is
run: is the blind spot a *real* additional cost on top of the
redundancy problem, or does redundancy alone already explain most of
uncertainty sampling's loss?

## 12. Statistical check and replication (added 2026-10-21)

When I went back and tested the win count properly, the original 13/30 turned out to be indistinguishable from a coin flip (exact two-sided binomial test, p = 0.59; 95% interval for the win rate 0.25 to 0.63). I reran the comparison unchanged, through the experiment's own `run_trial`, on 100 new trials whose seeds don't overlap the originals (`experiments/rq_replication.py`; per-trial results in `data/replication_results.json`). The criterion was written down before the rerun: an effect counts as replicated only if the new win rate lands on the same side of 50% as the original and a two-sided sign test on the new trials (ties dropped) gives p < 0.05.

**Result:** uncertainty sampling won 43 of 100 and lost 57 (no ties), sign test p = 0.19. The average final accuracy gap was 4.7 points in random sampling's favour.

**What this changes:** the pre-registered verdict above stands, since the prediction (entropy wins 60 to 70% of trials) is clearly wrong. But the stronger framing in Sections 7 to 9, that uncertainty sampling is worse than picking a random point, isn't supported. The honest summary is that uncertainty sampling showed no advantage over random sampling in this setting. The direction favoured random in both runs, but not reliably. The redundancy mechanism in Section 9 still explains why it fails to win; it doesn't show that it loses.
