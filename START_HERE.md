# Start here

A guide to this repo: what the project is, how I worked, what I found, and where everything lives. For the findings in essay form, read [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).

## The project in three lines

- **Question:** when can a probabilistic model's uncertainty be trusted as a signal for deciding what to observe next?
- **Laboratory:** Gaussian mixture models fitted with EM, implemented from scratch in `src/gmmlab/`.
- **What I was really practising:** the research loop end to end (question, prediction, experiment, result, explanation, next question) and owning every step of it.

The seven research questions are listed in `README.md`. Their numbering was locked at the start, and follow-ups were appended rather than renumbered.

## Reading order

1. [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for the whole story in one place.
2. The results table below, for the scoreboard.
3. Any `research_log/RQ*.md` file whose result surprises you. Each one has the prediction I committed to before running anything, the raw result, what I didn't expect, my best explanation, and its limitations.
4. `experiments/` for the script behind any number, and `src/gmmlab/` for the model itself.

## Results at a glance

Every prediction was written down, with a "this is wrong if..." line, before the experiment ran. I was wrong more often than right, which is the point of committing in advance.

The original runs used 30 trials. An exact binomial test afterwards showed that several of those win rates couldn't be told apart from a coin flip, so I reran those comparisons, plus RQ7, on 100 new trials with fresh seeds (`experiments/rq_replication.py`). An effect counts as replicated if it lands on the same side of 50% with sign-test p < 0.05. The "Verdict" column is the original pre-registered call, left as it was; the last column is what the evidence supports now.

| RQ | Question | What I predicted | Original result (30 trials) | Replication (100 new trials) | Verdict on the prediction | Where it stands now |
|---|---|---|---|---|---|---|
| RQ1 | Does cluster overlap raise the model's uncertainty in the ambiguous region? | Entropy rises toward its maximum as clusters overlap | Entropy fell to 0.000 at $c=1.0$ (Bayes-optimal: 0.684); EM found a lopsided fit instead | n/a | Wrong at $c \le 1$ | Holds (9 of 10 restarts, and every seed in a later 20-seed scan) |
| RQ2 | Is that lopsided fit common, and does k-means++ prevent it? | Most overlapping trials go lopsided; k-means++ may reduce it | 14 to 19 of 30 per overlapping condition, 0 of 20 at $c=4$; k-means++ slightly *more* lopsided | n/a | Partly right, partly wrong | Holds for overlap vs. no overlap; the k-means++ difference is small |
| RQ2 follow-up | Does k-means++'s second seed land inside the small component? | About 80% (wrong below 60%) | 19.4% (7/36), about 8x chance | n/a | Wrong | Holds |
| RQ3 | Does entropy sampling beat random sampling? | Entropy wins 60 to 70% | 13/30 (43.3%), p = 0.59 | 43/100, p = 0.19 | Wrong | **No advantage for entropy**, but no reliable loss either |
| RQ4 | Does sequential querying beat a one-shot top-10 batch? | Sequential wins 50 to 60% | 20/30 (66.7%), p = 0.10 | 56/100 (7 ties), p = 0.06 | Held | Suggestive, not established |
| RQ5a | Does a cheating "confidently wrong" oracle beat entropy? | Oracle wins | 12/30 (40.0%), p = 0.36 | 35/100, p = 0.005 | Wrong | **Replicated: the oracle is reliably worse** |
| RQ5b | Under a true 90/10 imbalance, does entropy beat random? | Entropy wins only 30 to 40% | 2/30 (6.7%), p < 0.001 | not rerun | Held | **Holds: entropy reliably worse** |
| RQ5c | With outliers disguised as boundary points? | Entropy wins under 40% | 6/30 (20.0%), p = 0.001 | not rerun | Held | **Holds: entropy reliably worse** |
| RQ5d | Fitting $K=2$ to 3 true clusters? | (a) 66.7% accuracy ceiling; (b) entropy wins 40 to 45% | (a) 66.8%; (b) 30/30 (100%), p < 0.001 | not rerun | (a) held, (b) wrong | Holds, though a metric artifact inflates (b) |
| RQ6 | Does restart disagreement beat entropy? | Wins 45 to 60% | 17/30 (56.7%), p = 0.58 | 67/100, p < 0.001 | Held | **Replicated, and stronger** |
| RQ6c | Does bootstrap disagreement beat entropy? | Wins no more than 50% | 18/30 (60.0%), p = 0.36 | 52/100, p = 0.48 | Wrong | **Did not replicate**: no reliable advantage, as first predicted |
| RQ7 | Does disagreement across samples from a real posterior beat entropy? | Wins 50 to 65% | 24/30 (80.0%), p = 0.001 | 59/100, p = 0.04 | Wrong, on the high side | **Replicated at 59%**, inside the original prediction; the 80% was an overestimate |

The thread that ties these together: uncertainty sampling never reliably beat random in the standard setting, lost reliably when the uncertainty came from imbalance or disguised outliers, and won when the useful information was a small, finite set of fixable points (RQ5d). Disagreement across restarts and across posterior samples both beat plain entropy reliably, while bootstrap disagreement didn't. And two explanations I wrote at 30 trials turned out to be explaining noise, which is the main reason I'd plan sample sizes up front next time.

## Bugs and detours worth knowing about

Some of the most useful lessons came from things going wrong. Each of these is written up in full in the relevant log:

- **RQ4: a smoke test that lied.** Running a script directly puts its own folder on the import path, not the project root, so `import experiments.rq3_entropy_vs_random` failed. My `python -c` smoke test had hidden this, because that invocation adds the current directory instead. Fixed with a small path bootstrap and re-checked with the real command.
- **RQ5c: decoys that broke EM itself.** Placing the disguised-outlier points too close together let EM fit a near-singular covariance around them as a group, which tripped my own "log-likelihood must never decrease" check through real floating-point precision loss. Fixed by spreading the decoys apart and stress-testing all 30 seeds before trusting the run.
- **RQ6: a tie-breaking bias.** The disagreement score had only a handful of possible values, so in one trial 317 of 600 points tied for the top score and `argmax` quietly picked by array position, which happened to correlate with the true label. Fixed by breaking ties with entropy.
- **RQ6: a precision failure in shared code.** One restart on ordinary data hit the same near-singular collapse and raised an error. I fixed it in `gmmlab.em.fit_many` (skip that restart, and raise if every restart fails) rather than patching around it locally, then confirmed RQ3's already-published numbers came out unchanged.
- **RQ5d: a metric that stops being honest.** "Accuracy on still-unlabeled points" inflates once some points are permanently unfixable, because querying one removes it from the grading pool without fixing it. Part of RQ5d's 100% is this artifact.
- **Thirty trials wasn't enough.** I only checked statistical significance after writing everything up, and found that five of the nine head-to-head comparisons were indistinguishable from a coin flip. The 100-trial replication confirmed three effects, left two inconclusive, and showed that bootstrap disagreement's apparent win, along with the explanation I'd written for it, was noise.
- **RQ7: the starting point was worse than I thought.** At the project's standard separation, every one of 20 scanned seeds gave an initial EM fit near chance accuracy. That doesn't change the head-to-head comparisons (both strategies always start from the same fit), but it changes how the starting point should be read. It's now listed under known gaps in the README.

## How I worked

### One rule

> **Never keep something I can't explain.**

AI could write code, but for anything in the repo I had to be able to answer six questions:

| # | Question | Example (Gaussian log-density) |
|---|---|---|
| 1 | What does it compute? | how likely a point is under one Gaussian |
| 2 | Why do we need it? | EM needs each component's likelihood |
| 3 | Which equation is it? | $\ln\mathcal{N}(\mathbf{x}\mid\boldsymbol\mu,\boldsymbol\Sigma)$ |
| 4 | What does it assume? | $\boldsymbol\Sigma$ is positive definite |
| 5 | How do we know it's right? | matches a hand calculation and `scipy` |
| 6 | What breaks it? | singular covariance, underflow |

`README.md` has a fuller note on how I used an AI assistant on this project and what I did myself.

### One mode at a time

Early on I was mixing learning theory, writing code, and running experiments all at once, and it got confusing fast. After that, every stage had exactly one mode:

| Mode | What it covers | What it doesn't |
|---|---|---|
| **LEARN** | the math: intuition, derivation, answering questions, correcting mistakes | writing code, designing experiments |
| **BUILD** | implementing one function at a time, explaining it, testing it | running research questions |
| **INVESTIGATE** | one research question at a time, through the full loop | learning new theory (that means stopping for a LEARN stage) |

Ideas for later questions went into `research_log/IDEAS.md` so they didn't derail the current stage.

### House rules

The logs refer to these by number:

1. LEARN stages go intuition, then formulas, then derivation, then my own answers, then corrections with a worked answer and the reason.
2. No code file before its math is understood.
3. Every experiment gets a research-log entry, and "Unexpected results" and "Next question" are mandatory.
4. From RQ3 on, the analysis plan goes into the log before anything runs.
5. Uncertainty sampling is allowed to lose to random. That is a result.
6. Scope changes get written down with a reason.

## Roadmap

| Stage | Mode | What | Finished when I could... | Status |
|---|---|---|---|---|
| 0 | n/a | Freeze the question and the RQ list | (in README) | Done |
| 1 | LEARN | Latent variables, GMMs, responsibilities, entropy (Lesson 1) | explain the generative story and compute a responsibility by hand | Done |
| 2 | LEARN | EM theory: Jensen, the ELBO, the E-step (Lesson 2), then the M-step and Lagrange multipliers (Lesson 3) | derive the E-step and M-step without notes | Done |
| 3 | BUILD | GMM/EM from scratch, one function at a time | answer the six questions for each function | Done |
| 4 | BUILD | Sanity checks: easy clusters, monotone log-likelihood, comparison with scikit-learn | recover an easy dataset and read the figure myself | Done |
| 5 | INVESTIGATE | RQ1, cluster overlap | write a log entry with an unexpected result and a next question | Done |
| 6 | INVESTIGATE | RQ2, initialization, plus its mechanism follow-up | same | Done |
| 7 | LEARN | Clamped (semi-supervised) EM and experimental design | explain what a revealed label does to the model, and state an analysis plan before running anything | Done |
| 8 | INVESTIGATE | RQ3 to RQ5 (entropy vs. random, batching, four failure modes) | same | Done |
| 9 | INVESTIGATE | RQ6 and RQ7, then the technical report | same | Done |

The work ran from mid-September to late October 2026. RQ1 started on September 13, and RQ7 finished on October 21.

## Where things are

| Path | What |
|---|---|
| `TECHNICAL_REPORT.md` | the findings, written up as one report |
| `README.md` | project description, the locked RQ list, known gaps, how AI was used |
| `research_log/` | one file per research question, using `00_TEMPLATE.md`, plus `IDEAS.md` |
| `src/gmmlab/` | the from-scratch GMM/EM implementation |
| `experiments/` | the script behind every result |
| `tests/` | 76 tests for `gmmlab`, including checks against scikit-learn |
| `figures/` | every figure the experiments produce |
| `lessons/` | the notes and exercises I worked through while learning the math, written with an AI tutor; its feedback on my answers is in the `*_review.md` files |

## Reproducing results

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q
.venv/bin/python experiments/rq3_entropy_vs_random.py
```

Every experiment script runs the same way and regenerates its figure in `figures/`. Most finish in a few minutes; the slowest (RQ7 at about ten minutes, RQ6c at about forty) refit an ensemble at every query step.
