# RQ6c: Do bootstrap ensembles provide information beyond responsibility entropy?

- **Date started / run:** 2026-10-20
- **Status:** done. Hypothesis wrong: bootstrap-disagreement beat entropy,
  it did not stay at or below chance level.
- **Replication (added 2026-10-21):** Did not replicate at 100 new trials (52/100, p = 0.48); the interpretation below is withdrawn, see Section 12.
- **Code:** `experiments/rq6c_bootstrap_disagreement.py` · **Figure:**
  `figures/rq6c_bootstrap_disagreement.png`
- **Builds directly on:** `research_log/RQ6_restart_disagreement.md` (the
  restart-ensemble half of this same locked RQ, reported separately per
  README's own wording: *"Restarts measure optimization uncertainty;
  bootstraps measure data-sampling uncertainty. They are reported
  separately, and neither is called 'epistemic' without qualification."*).

## Plain-English summary (read this first; Sections 1–6 below are the technical record)

RQ6 tested restart-disagreement (same data, different starting points) and
found it beats both plain entropy and RQ5a's cheating oracle, since its
picks carry less leverage than the oracle's always-extreme targets. This asks
the other half of the locked question: does **bootstrap-disagreement**
(different resampled data, via drawing with replacement) behave the same
way, or does varying the *data* instead of the *starting point* produce a
different signal?

Worked out, not asserted, before committing to a hypothesis: the
collapse pathology (RQ1) is caused by a tiny, tight handful of points (3-5,
out of 600) sitting unusually close together. The standard "63.2% bootstrap
rule" says any specific point has about a 37% chance of being left out of
one resample entirely, so the chance all 4 points in a small collapse
cluster survive a given resample is only $0.63^4\approx16\%$, meaning
roughly an 84% chance at least one of them gets dropped. A handful of
points is exactly the kind of structure bootstrap resampling is likely to
disrupt: this cluster's existence should be *fragile* under resampling,
unlike restart ensembles, where the same full dataset is reused every time
and only the optimizer's path differs.

**The hypothesis was wrong: bootstrap-disagreement beat entropy in 18/30
trials (60.0%), not at or below chance.** It even slightly outperformed
restart-disagreement's 56.7% (RQ6). The predicted mechanism (fragility
under resampling) was correctly reasoned statistically, but the
*conclusion* drawn from it was backwards: fragility doesn't make the
signal noisy and unreliable, it makes it a direct, targeted detector for
exactly the vulnerable spot. A cluster that falls apart under small data
perturbations is precisely the thing worth flagging, and bootstrap
resampling tests exactly that, by perturbing the data itself, every single
ensemble member.

## 1. Research question

(Locked wording, README, bootstrap half.) Do bootstrap ensembles provide
information beyond responsibility entropy? Concretely: does querying by
**disagreement across 5 fits on independent bootstrap resamples** do
better, worse, or about the same as plain entropy sampling, and how does
it compare to restart-disagreement's confirmed 56.7% win rate (RQ6)?

## 2. Hypothesis

**Committed on 2026-10-20, before any code was written:**

> Bootstrap-disagreement sampling's win rate against plain entropy will not
> exceed 50% across 30 trials. **Wrong if it exceeds 50%.**

My reasoning at the time, condensed: data fragility under
resampling should make bootstrap disagreement a noisier, less selective
signal than restart disagreement, not necessarily lower-leverage in the
same clean way restart's was, just less reliably informative, pulling its
performance down toward chance level rather than up toward restart's
56.7%.

## 3. Experimental setup

Same $c=1.0$, $N=600$, initial `fit_best_of`, warm-started refits
($n\_iter=200$), budget $T=60$, 30 trials, held-out accuracy, identical
setup to RQ3/RQ6, for direct comparability.

**New strategy, "bootstrap-disagreement":** at each query step, draw 5
bootstrap resamples of the *current* dataset (with replacement, same size
as the original; each original point's own clamp status, if any, carries
over into however many copies of it land in a given resample). Fit EM
fresh on each resample (`init="random"`, independent seeds, $n\_iter=500$,
matching restart-disagreement's ensemble settings for direct
comparability). Evaluate each bootstrap-fitted model's responsibilities on
the **original** (not resampled) points, so every candidate gets a
prediction from every ensemble member regardless of whether it happened to
appear in that particular resample. Component labeling realigned to the
main fit via `match_components`, same necessity as RQ6 (independently-fit
models have no reason to agree on which physical cluster is "component
0"). Disagreement and tie-breaking computed identically to
restart-disagreement (vote-count disagreement, entropy as tie-break among
maximally-disagreeing candidates), reusing the same fix from RQ6, Section 8, not
re-deriving it.

## 4. Independent variable(s)

Strategy $\in\{$bootstrap-disagreement, entropy$\}$. $c=1.0$, $T=60$,
ensemble size (5 resamples per step) held fixed (not swept).

## 5. Dependent variable(s) / metric

Held-out accuracy on still-unlabeled points, same convention as RQ3/RQ6.
Win/loss count at the final budget.

## 6. Controls

- **Common random numbers:** both strategies run on the same 30 datasets,
  same initial fit, same seeds as RQ6/RQ6b for direct comparability
  across all three (restart, bootstrap, redundancy-check) results.
- Component-label realignment applied every step, same as RQ6.
- $N=600$, $K=2$, $\sigma=1$, `reg_covar`, EM's `tol` held fixed.
- Reuses `gmmlab.em.fit_many`'s robustness fix (RQ6, Section 8) automatically,
  since it's shared library code: a restart-of-a-bootstrap-resample that
  hits the same rare floating-point edge case won't crash the ensemble.

## 7. Results

Full output: `experiments/rq6c_bootstrap_disagreement.py`; figure:
`figures/rq6c_bootstrap_disagreement.png`.

| Quantity | Value |
|---|---|
| Bootstrap-disagreement beats entropy | 18/30 = 60.0% |
| RQ6's restart-disagreement, for comparison | 56.7% |
| Predicted | not exceed 50% |
| Pre-committed "wrong" line | exceeds 50% |
| **Verdict** | **Hypothesis wrong** (60.0% > 50%) |

## 8. Unexpected results

The direction, not just the margin. The prediction wasn't "bootstrap wins
by a smaller margin than restart"; it was "bootstrap should not beat
entropy at all." It won more often than entropy *and* more often than
restart-disagreement did. The specific reasoning behind the prediction
(data fragility under resampling) was statistically sound (the 63.2%
bootstrap rule does mean small clusters are vulnerable to being
disrupted), but the inference drawn from that fact (fragile therefore
noisy therefore less useful) was the wrong direction.

## 9. Interpretation

Fragility isn't noise here, it's the signal. The collapse-triggering
cluster (RQ1) is small and tight specifically *because* it's an unusual,
localized coincidence in the data, not a robust feature of the true
underlying structure. That same smallness is exactly what makes it
vulnerable to bootstrap resampling knocking it out. So when 5 bootstrap
fits disagree sharply about a point, that disagreement is pointing
at the fragile, pathological region, not at generic noise
spread evenly across the dataset. Restart-disagreement finds this same
kind of region a different way (via optimizer multi-modality on fixed
data); bootstrap-disagreement finds it by directly testing whether the
region survives small perturbations to the data itself. Different
mechanisms, similar practical effect: both produce a *softer* signal than
RQ5a's oracle (no single model needs to be extremely certain), which is
the throughline connecting all three ensemble-based strategies' escape
from the oracle's overcorrection problem.

## 10. Limitations

- Ensemble size fixed at 5, not swept, same as RQ6.
- Whether bootstrap-disagreement also escapes RQ3's redundancy problem
  (the way RQ6, Section 12 checked for restart-disagreement) was not tested here,
  an open, analogous follow-up.
- Computationally the most expensive RQ yet (5 fresh fits on resampled
  data per step, on top of the main warm-started fit, ~81s/trial).
- The "fragility as targeted signal" interpretation (Section 9) is a plausible,
  mechanism-consistent story, not independently verified by, e.g.,
  checking whether bootstrap-disagreement's highest-scoring points
  coincide with the collapse-triggering cluster's members.

## 11. Next question

**Primary:** the same redundancy-spread check RQ6, Section 12 ran for
restart-disagreement, run for bootstrap-disagreement: does its query
spread also show the same "wide in a minority of trials, similar-to-entropy
in most" pattern, or something different, given its mechanism (data
fragility) is different from restart's (optimizer multi-modality)?

**Secondary:** verify Section 9's "fragility as targeted signal" story:
check whether bootstrap-disagreement's top-scoring points, early in a
query loop, coincide with the specific small cluster driving that
trial's collapse pathology, the same way RQ6's diagnostic concretely
traced restart-disagreement's mechanism rather than just asserting it.

## 12. Statistical check and replication (added 2026-10-21)

The original 18/30 wasn't significant on its own (exact two-sided binomial test, p = 0.36; 95% interval 0.41 to 0.77). I reran the comparison unchanged, through the experiment's own `run_trial`, on 100 new trials whose seeds don't overlap the originals (`experiments/rq_replication.py`; per-trial results in `data/replication_results.json`). The criterion was written down before the rerun: an effect counts as replicated only if the new win rate lands on the same side of 50% as the original and a two-sided sign test on the new trials (ties dropped) gives p < 0.05.

**Result:** bootstrap disagreement won 52, lost 44, and tied 4 (sign test p = 0.48). The average accuracy gap was 0.6 points.

**What this changes:** this did not replicate. The pre-registered verdict (wrong) was correct on its own terms at 30 trials, but the 60% now looks like noise: across 100 fresh trials bootstrap disagreement showed no reliable advantage over entropy, which is what my original hypothesis predicted. The interpretation in Section 9 (fragility as a targeted signal) and the "correction" described in the plain-English summary were explanations written for an effect that doesn't hold up, so I'm withdrawing them. I've left them in place above as a record of what I thought at the time; this section supersedes them.
