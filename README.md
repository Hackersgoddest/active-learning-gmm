# When Is Uncertainty Informative?

Investigating uncertainty and information acquisition in latent-variable models,
with Gaussian mixture models (GMMs) as the laboratory. GMM is the experimental
setting, not the research identity.

For what I found, read [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md). For a guide
to the repo and a one-table summary of every result, open
[START_HERE.md](START_HERE.md).

A self-directed research-training project. The goal is to practice the research
loop end to end: formulate questions, make predictions, implement the method,
run controlled experiments, investigate failures, replicate selected results,
and limit conclusions to what the evidence supports.

The project is deliberately small and controlled. It is not a polished library,
a publication, or a claim of a novel scientific contribution.

## Project status

The experiments use small synthetic settings so that the mathematics and
mechanisms remain inspectable. The results should be read as lessons from this
controlled laboratory, not as general claims about active learning or GMMs.

## Quick start

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q
.venv/bin/python experiments/rq3_entropy_vs_random.py
```

The test suite validates the core implementation. The experiment command
regenerates the corresponding figure in `figures/`. The longer experiments,
especially RQ6c and RQ7, may take substantially longer.

> **Question → Hypothesis → Experiment → Result → Explanation → New Question**

The detailed research logs in `research_log/` preserve that process.

## Central question (kept neutral on purpose)

> When can a probabilistic model's uncertainty be trusted as a signal for
> deciding what to observe next, and when does it stop being informative?

"Uncertain points teach the model more" is the **hypothesis**, not the claim.
Uncertainty is not the same as informativeness: a model can be uncertain because
a point lies between clusters, or because its parameters are poor, the model is
misspecified, the point is an outlier, or data are scarce.

**Uncertainty sampling is allowed to lose to random selection.** If it does, the
next question is *why*, and that is a result, not a failure.

## Research questions (numbering is locked; new questions are appended, never renumbered)

| RQ  | Question | Tier |
|-----|----------|------|
| RQ1 | How does cluster overlap affect EM's recovery of latent structure and its assignment uncertainty? | A |
| RQ2 | How sensitive is EM to initialization, and how does that interact with overlap? | A |
| RQ3 | Under what conditions does responsibility entropy improve learning efficiency relative to random selection when labels are limited? | A |
| RQ4 | Does top-k entropy produce an informative batch, or a redundant one? | A |
| RQ5 | Where does uncertainty fail: outliers/density, misspecification, imbalance? | A |
| RQ6 | Do bootstrap and restart ensembles provide information beyond responsibility entropy? (Restarts measure optimization uncertainty; bootstraps measure data-sampling uncertainty. They are reported separately, and neither is called "epistemic" without qualification.) | B |
| RQ7 | What changes when we place a posterior distribution over model parameters? | C |

I finished Tier A before touching Tiers B and C. All seven questions now have a
committed prediction, a real run, and a written-up result in `research_log/`.

## Authorship and AI assistance

This was a self-directed project completed with AI assistance. I selected the
central question and research questions, chose the order of investigation,
committed to predictions before result-producing runs, worked through the
mathematical material, interpreted the results, decided which follow-up
questions to pursue, and documented the limitations.

The AI assistant (Claude, through Claude Code) acted as a tutor and pair
programmer. It helped explain mathematics, generate and revise implementation
drafts, run experiments, and edit documentation. I reviewed and tested the
resulting work, and I do not claim authorship of code or explanations that I
cannot personally explain.

The project is presented as evidence of research preparation and persistence,
not as a claim of a novel scientific contribution.

## Known gaps, and where each gets closed

- **Singularities:** a full covariance can collapse onto one point (likelihood
  goes to infinity). `reg_covar` (Stage 3) stops that at the single-fit level,
  bounding the magnitude but not fixing the pathology itself. It's **not** a
  complete fix: found while designing RQ1 that a *near*-singular collapse (one
  component on a handful of points) can still out-score the honest fit on raw
  log-likelihood, so "pick the best of N restarts" needs a second guard.
  `gmmlab.em.is_degenerate` + `fit_best_of`'s filtering (Stage 5) is that
  guard. Numbers in `research_log/IDEAS.md`.
- **Log-sum-exp:** numerically safe responsibilities (Stage 3).
- **Label switching:** resolved. `gmmlab.matching.match_components` finds the
  optimal correspondence between two fits' components, using the Hungarian
  algorithm rather than nearest-neighbor greedy matching, which can be
  meaningfully suboptimal (see `tests/test_matching.py`). Used by the RQ1
  accuracy metric and by Stage 4's gmmlab-vs-sklearn comparisons.
- **Choosing K:** we assume K is known; a wrong K is a form of misspecification
  (RQ5).
- **Initialization methods** (random vs. k-means++): before RQ2. Stage 3's
  `em.fit` uses a simple baseline (K distinct data points as starting means).
  `gmmlab.em.fit_best_of` (best of N random restarts) is the practical
  mitigation used where a single unlucky seed would otherwise be mistaken for
  a bug (Stage 4), but it isn't a substitute for RQ2's systematic study of the
  sensitivity itself. Two concrete, reproducible examples of that sensitivity
  are already recorded in `research_log/IDEAS.md`.
- **What a revealed label does to the model** (clamped EM): Stage 7.
- **Experimental design** (metric, seeds, pre-registered analysis): Stage 7.
- **Which covariance model to fit:** resolved in Lesson 3. `gmmlab` fits
  `full` covariance. RQ1's data is generated `spherical`-`tied` (a special
  case), so recovery is a genuine test, not an easy one; `diag`/`spherical`/
  `tied` remain cheap one-line variants if a later RQ needs them.
- **Does an ensemble-disagreement strategy also escape the redundancy
  problem (RQ3/RQ4), not just the leverage/overcorrection problem (RQ5a)?**
  Written down per house rule 6 (scope changes get a reason): this question
  is not part of RQ6's original locked wording above, it emerged directly
  from RQ6's own diagnostic, asking why restart-disagreement showed
  widespread disagreement at all. Tested for restart-disagreement only so
  far (`research_log/RQ6_restart_disagreement.md`, Section 12): a real but
  trial-dependent effect, concentrated in a minority of trials, not the
  main explanation for its win. The leverage/overcorrection story remains
  the more consistent one. Bootstrap disagreement's analogous check was
  planned, but its win didn't replicate (see the next item), so there's no
  effect left to explain.
- **Statistical power of the original 30-trial runs.** I tested significance
  only after writing the results up: five of the nine head-to-head
  comparisons (RQ3, RQ4, RQ5a, RQ6, RQ6c) were indistinguishable from a coin
  flip at 30 trials. Closed by `experiments/rq_replication.py`, which reran
  those five plus RQ7 on 100 fresh trials each, with a replication criterion
  written down beforehand. RQ5a, RQ6 and RQ7 replicated; RQ3 and RQ4 stayed
  inconclusive; RQ6c's apparent win turned out to be noise. Each affected
  log has a section reporting this, and the original pre-registered verdicts
  are left unchanged. Still open: RQ5b to RQ5d were significant at 30 trials
  but haven't been rerun at the larger scale.
- **How common is the near-singular collapse at c=1.0, really?** Found
  while picking a validation seed for RQ7, not from a systematic sweep like
  RQ2's: scanned 20 data seeds through the exact `fit_best_of` call
  RQ3-RQ6 use, and every single one converged to an imbalanced initial fit
  (minority weight 2.7%-42%, accuracy against true labels 0.49-0.64,
  consistently near chance). Previously known to be *possible* (RQ1); this
  is the first time it was checked across many seeds in one place and found
  to be the norm, not an occasional unlucky draw. Not a bug in RQ3-RQ6,
  since both compared strategies always share that same starting point, so
  their head-to-head results are unaffected, but worth recording precisely
  because it changes how "the initial unsupervised fit" should be read
  going forward: at c=1.0, assume it starts near chance unless checked
  otherwise. Details: `research_log/RQ7_mcmc_and_vi.md`, Sections 7(f), 8.1, and 10.

## Environment

Per-project virtual environment in `.venv/` (numpy, scipy, matplotlib, pytest).
Editable install: `pip install -e ".[dev]"` so `import gmmlab` works anywhere.

## License

Code and documentation are released under the [MIT License](LICENSE).
