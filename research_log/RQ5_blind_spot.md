# RQ5: Where does uncertainty fail: outliers/density, misspecification, imbalance?

- **Date started / run:** 2026-10-14 to 2026-10-15
- **Status:** done. Hypothesis wrong, the opposite direction from predicted.
  First pass scoped to one angle only (see below); the other two
  (outliers/density, misspecification) are deliberately deferred, not
  forgotten.
- **Replication (added 2026-10-21):** Replicated at 100 new trials (oracle won 35/100, p = 0.005); see Section 12.
- **Code:** `experiments/rq5_blind_spot.py` (plus an additive
  `"oracle_wrong"` strategy branch added to
  `experiments/rq3_entropy_vs_random.py`'s `run_query_loop`, verified not to
  change RQ3/RQ4's existing results) · **Figure:**
  `figures/rq5_blind_spot.png`
- **Builds directly on:** `research_log/RQ3_entropy_vs_random.md` Section 11's
  "Secondary" next question (is the confidently-wrong blind spot a real,
  additional cost beyond the general redundancy problem, or does redundancy
  alone already explain uncertainty sampling's loss?) and
  `lessons/04_clamped_em_and_active_learning.md` Section 4.4.

## Plain-English summary (read this first; Sections 1–6 below are the technical record)

RQ3 found that uncertainty sampling loses to random sampling, and that the
loss looked similar whether the trial started lopsided or balanced, which
muddied whether the specific confidently-wrong blind spot (Lesson 4, Section 4.4)
was costing anything, versus the more general redundancy problem
(RQ3/RQ4) being the whole story on its own.

This test isolates the blind spot: a deliberately unrealistic
**"confidently-wrong oracle"** strategy that queries whichever still-
unlabeled point the model is currently most confident about *and wrong*
about (checking against the true label, something no real system could
do; this is a diagnostic ceiling, not a deployable strategy). If it clearly
beats real uncertainty sampling on severely-lopsided starts, the blind spot
is real and costly. If it doesn't, redundancy alone was already the story.

A real risk was checked *before* committing to this design: would the
oracle suffer the same "stuck in a narrow neighborhood" redundancy problem
uncertainty sampling had? A direct trace (10 steps, one trial) showed no:
the oracle's picks were scattered across wildly different positions, and
`pi` (the model's belief about each cluster's size) swung dramatically each
step (0.034 → 0.019 → 0.355 → 0.550 → 0.864 ...), unlike uncertainty
sampling's slow creep. Confidently-wrong points have leverage (Lesson 4, Section 4.3) close to the theoretical maximum, which is the likely reason: each
reveal shifts the model's whole picture enough that the ranking can't get
stuck. A new, unplanned observation from that same trace: the
correction **overshoots and oscillates** rather than settling smoothly;
worth watching in the full run.

**It mattered more than expected: the oracle lost.** Across 30 severely
lopsided trials, the oracle beat real uncertainty sampling in only 12
(40.0%), below the pre-committed "wrong" line, and the *opposite* direction
from what leverage alone would predict. Combined with RQ3 (entropy lost to
random) and RQ4 (batch entropy lost to sequential entropy), this is now a
third result pointing the same way: **a point's leverage over the model's
parameters doesn't guarantee that correcting it teaches the model something
useful.** The missing piece, worked out from the `pi` trace: clamping is an
all-or-nothing, maximum-strength correction, and a high-leverage point's
full-strength correction can overshoot past the right answer and flip many
*other* points' classifications the other way, spending later queries
fixing a self-inflicted overcorrection instead of making new progress.

## 1. Research question

(Scoped sub-question of the locked RQ5.) On trials where the initial fit is
severely lopsided, does a strategy that can directly target
confidently-wrong points outperform real uncertainty sampling, i.e., is the
blind spot (Lesson 4, Section 4.4) a real, additional cost, not just a theoretical
possibility?

## 2. Hypothesis

**Committed on 2026-10-14, before any code was written:**

> The confidently-wrong oracle beats real uncertainty sampling (held-out
> accuracy at the same budget) in 60-70% of 30 severely-lopsided-start
> trials. **Wrong if it wins in fewer than 50% of trials.**

## 3. Experimental setup

Same $c=1.0$, $N=600$, initial `fit_best_of` ($R=10$, $n\_iter=2000$), warm-
started refits ($n\_iter=200$), and budget $T=60$ as RQ3/RQ4. New pieces:

- **Trial selection, not just reuse of RQ3's 30 seeds:** scan candidate trial
  indices (same seed scheme as RQ3: `data_seed = idx`, `restart_seed =
  idx + 10_000`) starting from 0, keep only those whose *initial* fit is
  severely lopsided ($\min_k\pi_k < 0.10$, RQ2/RQ3's own threshold), until 30
  qualifying trials are collected. RQ3 found this threshold catches roughly
  half of trials at $c=1.0$, so this scan is expected to need on the order of
  60 candidates to find 30, not a different population, just a filtered
  subset of the same one.
- **New strategy, added to `rq3.run_query_loop` directly (not
  reimplemented):** `"oracle_wrong"`: among still-unlabeled points predicted
  incorrectly (checked against `z_true`, only possible because this is
  synthetic data), pick the one with the lowest entropy (i.e. most
  confident). If no currently-wrong points remain, fall back to highest
  entropy among the rest (shouldn't matter in practice, since a severely lopsided
  start has hundreds of wrong points available).
- Compared against real uncertainty sampling (`rq3.run_query_loop(...,
  "uncertainty", None)`, unchanged) on the *same* 30 filtered trials.
- Both strategies are deterministic given the shared initial fit; no
  sub-repeats needed for either (same reasoning as RQ3's sequential
  condition).

## 4. Independent variable(s)

Strategy $\in\{$oracle\_wrong, uncertainty$\}$. $c=1.0$, $T=60$ held fixed.
Trial population restricted to severely-lopsided starts only (not a sweep;
a deliberate filter, see Section 3).

## 5. Dependent variable(s) / metric

Same as RQ3/RQ4: accuracy on the still-unlabeled points only, compared at
the final budget for the win/loss count, plotted across the budget for the
figure.

## 6. Controls

- **Common random numbers:** both strategies run on the same filtered
  trials, starting from the same initial fit each time.
- **Same code path as RQ3/RQ4** (`run_query_loop`, with one additive
  strategy branch), not a parallel reimplementation that could silently
  drift from the already-verified sequential logic.
- $N=600$, $K=2$, $\sigma=1$, `reg_covar`, EM's `tol` held fixed, matching
  RQ1-RQ4.

## 7. Results

Full output: `experiments/rq5_blind_spot.py` (scanned 56 candidate trials to
find 30 severely lopsided ones, matching RQ3's own observation that roughly
half of $c=1.0$ trials land there); figure: `figures/rq5_blind_spot.png`.

| Quantity | Value |
|---|---|
| Oracle beats real uncertainty sampling | 12/30 = 40.0% |
| Predicted | 60-70% |
| Pre-committed "wrong" line | below 50% |
| **Verdict** | **Hypothesis wrong** (40.0% < 50%, opposite direction) |

## 8. Unexpected results

**Not a near-miss: the wrong direction entirely.** The oracle, which
*cheats* by checking the true label and specifically targets the single
most damaging error in the model, loses to plain entropy-based uncertainty
sampling more often than it wins. Leverage (Lesson 4, Section 4.3) and correctness
both favored the oracle; neither was enough.

This is the third result in a row pointing the same way: RQ3 (real entropy
sampling lost to random), RQ4 (batch entropy lost to sequential entropy),
and now RQ5 (even a cheating, maximally-informed oracle loses to plain
entropy sampling). None of these are about whether a point is "correct" or
"informative" in isolation; each one is about *how the correction gets
used over a budget*, a different axis entirely from how good any single
pick looks.

## 9. Interpretation

The mechanism, worked out from the `pi` trace recorded in the
summary above: clamping is a full-strength, all-or-nothing correction: a
point's responsibility jumps straight to $(1,0)$ or $(0,1)$, no partial
step. For a confidently-wrong point, whose leverage is close to the
theoretical maximum (Lesson 4, Section 4.3's $1-p$), that full-strength correction
doesn't just fix the one point, because $\boldsymbol\mu_k,\boldsymbol\Sigma_k$
are weighted averages over every point (Section 4.2's A/B/C example), one such
clamp can swing the model's whole picture far enough to **overshoot past
balanced and into being wrong the other way**, exactly what the observed
`pi` sequence shows ($0.034\to0.019\to0.355\to0.550\to0.864\to0.833\to
0.718\to0.515\to0.344\to0.321\to0.535$, crossing 0.5 repeatedly, not
settling near it). Each overshoot then costs some of the remaining budget
correcting a problem the strategy's *own* previous query caused, rather than
making new progress.

Entropy sampling's points sit near $(0.5,0.5)$, modest leverage, so each
clamp nudges the model gently. Slow and redundant (RQ3/RQ4's cost), but it
doesn't blow past the target and have to double back. Put together with
RQ3/RQ4: there appears to be a real tradeoff between how much a point's
correction teaches the model (leverage) and how *stable* that correction is
(risk of overshoot): low-leverage points are stable but redundant,
high-leverage points are non-redundant but can destabilize the fit. Neither
extreme tested so far (RQ3's entropy, RQ5's oracle) is doing well; this
hasn't yet tested anything in between.

## 10. Limitations

- This only tests the confidently-wrong angle of RQ5's three named failure
  modes (outliers/density and misspecification are deliberately deferred,
  header note).
- The oracle is unrealistic by construction (requires ground truth); a
  real system would need a *proxy* for "confidently wrong" (e.g.
  disagreement across restarts or bootstraps, RQ6's question) to capture
  any of this value, and that proxy might behave differently from the
  ground-truth oracle tested here.
- The overshoot mechanism (Section 9) was diagnosed from a trace and a correlation
  with the final win/loss count, not measured directly as "how much budget
  does overshoot-correction consume." That's a direct, testable follow-up,
  not yet done.
- 30 trials, one comparison each, real direction, not a high-precision
  estimate of the size of the effect.

## 11. Next question

**Primary, motivated by Section 9:** does *damping* the correction (e.g.
clamping to a strong-but-not-one-hot responsibility, say $(0.9, 0.1)$
instead of $(1, 0)$) reduce the overshoot while keeping most of the
leverage advantage? This would test whether the problem is
"high-leverage points are bad to query" (in which case damping wouldn't
help much) or "full-strength corrections are too aggressive regardless of
which point" (in which case damping should help).

**Secondary:** RQ6 (do restart/bootstrap ensembles carry information beyond
responsibility entropy?) was motivated under the assumption that a realistic
proxy for the oracle's target would be valuable. Given Sections 7-9, that
motivation needs revisiting first: a realistic proxy for "confidently wrong"
may inherit the *same* overshoot problem the oracle has, not just the
oracle's blind-spot advantage. Worth deciding explicitly before investing in
RQ6, rather than assuming the oracle's failure doesn't apply to it.

## 12. Statistical check and replication (added 2026-10-21)

The original 12/30 wasn't significant on its own (exact two-sided binomial test, p = 0.36; 95% interval 0.23 to 0.59). I reran the comparison unchanged, through the experiment's own `run_trial`, on 100 new trials whose seeds don't overlap the originals (`experiments/rq_replication.py`; per-trial results in `data/replication_results.json`). The criterion was written down before the rerun: an effect counts as replicated only if the new win rate lands on the same side of 50% as the original and a two-sided sign test on the new trials (ties dropped) gives p < 0.05. New trials were chosen the same way as the originals (severely lopsided starts only), scanning candidate datasets from index 1000 upwards; the first 100 lopsided ones were found by candidate 1191.

**Result:** the oracle won 35, lost 64, and tied 1 (sign test p = 0.005). The average accuracy gap was 5.0 points in plain uncertainty sampling's favour.

**What this changes:** replicated. The oracle really is worse than plain uncertainty sampling, so the overcorrection story in Section 9 now rests on a reliable effect rather than a single 30-trial run.
