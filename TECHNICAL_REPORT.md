# When Is Uncertainty Informative? A Study of Active Learning in Gaussian Mixture Models

Frank Konadu Oppong

## Abstract

This report summarizes a self-directed research-training project built around one
question: when can a probabilistic model's uncertainty be trusted as a signal for
deciding what to label next, and when does it stop being trustworthy? I used
Gaussian mixture models fitted with EM, implemented from scratch, as the
laboratory. Across seven research questions, several follow-ups, and larger
follow-up runs for selected comparisons, some effects replicated and others
remained inconclusive. Uncertainty sampling showed no advantage over random
sampling in the standard overlapping setting, lost reliably to random under true
class imbalance and when outliers were disguised as boundary points, and won
every trial when the useful information was a small, finite set of fixable
points. A "cheating" oracle that always queries the model's most confidently
wrong point did reliably *worse* than plain uncertainty sampling, because a
full-strength correction to a maximally wrong point overshoots. Disagreement
across random restarts and disagreement across samples from a Bayesian posterior
over the cluster means both beat plain uncertainty sampling in the replicated
setting, while bootstrap disagreement did not. Along the way I found a
reproducible failure mode in which EM's fitted model is most confident exactly
where it is wrong, because an unbalanced, partially collapsed solution can beat
the honest one on raw log-likelihood. This is a controlled research-training
study, not a claim of general performance across active-learning systems or a
novel scientific contribution.

## 1. Motivation

Active learning tools often use model uncertainty to decide what to label next, on the implicit assumption that uncertain points are the most useful ones to learn from. That assumption is not automatically true. A model can be uncertain about a point because the point is genuinely ambiguous, or because the model's own parameters are still wrong, or because the point is an outlier the model can't explain well under any hypothesis, or because the model is simply the wrong shape for the data. Each of those is a different failure mode with a different fix, and conflating them means trusting uncertainty in cases where it shouldn't be trusted.

I picked Gaussian mixture models fitted with EM as the setting because the mechanism is small enough to fully derive and implement by hand, but rich enough to produce real, non-obvious failure modes. The core GMM/EM machinery was implemented explicitly rather than treated as a
black box, and its behavior was checked against scikit-learn. This made it
possible to inspect the mechanisms behind the results and work through the
relevant mathematics directly.

## 2. Methods

### 2.1 The model

A two-component Gaussian mixture is fit to data with the standard EM algorithm: an E-step computes each point's responsibility (soft probability of belonging to each component) and an M-step updates the mixing weights, means, and covariances as responsibility-weighted statistics. The project implements the Gaussian log-density, a numerically stable log-sum-exp, the E-step, the M-step, the full EM loop, and a Hungarian-algorithm-based component matcher (needed any time two independently fitted models need to be compared, since nothing forces "component 0" to mean the same thing twice). All of it was checked against `scikit-learn` on both easy and hard synthetic cases before being trusted for anything downstream.

A recurring structural fact worth stating up front, because it explains several results below: a full-covariance Gaussian mixture's log-likelihood is unbounded above. A component is always free to shrink onto a small, tight handful of points and claim arbitrarily high density there. `reg_covar` keeps this from reaching literal infinity, but a near-singular shrink can still beat an honest, balanced fit on raw log-likelihood by a wide margin. This single fact turned out to drive a surprising amount of what follows.

### 2.2 Active learning setup

Most research questions compare query strategies in the same semi-supervised loop: start from an unsupervised EM fit, then repeatedly pick one unlabeled point, reveal its true label, clamp its responsibility to a one-hot vector at that label, and refit (warm-started from the previous parameters). Because every revealed point's label feeds into the same shared mean, variance, and mixing weight through the M-step, clamping one point reshapes the whole model, not just that point's own classification. The metric throughout is held-out accuracy: the fraction of still-*unlabeled* points the model classifies correctly after each query, never grading the model on labels it has already been handed.

### 2.3 Experimental discipline

A few rules were followed throughout, mostly because the project kept finding ways to fool itself without them: a numeric hypothesis, including a specific "I was wrong if..." line, was committed before any result-producing run, never after; both strategies being compared always ran on the same datasets with the same random seeds (common random numbers), so a win or loss reflects the strategy and not an unlucky draw; and every experiment was written up with its unexpected results and next question recorded honestly, including every time the pre-committed hypothesis turned out to be wrong.

### 2.4 Statistics, and a replication I added afterwards

Every head-to-head comparison is scored per trial: strategy A "wins" a trial if its final held-out accuracy beats strategy B's on the same dataset. With 30 trials per comparison, I report an exact two-sided binomial test of the win count against 50%, with a 95% Clopper-Pearson interval on the win rate. Checking this after the fact showed that 30 trials only had enough power for large effects: five of the nine original comparisons could not be told apart from a coin flip. So I reran those five, plus RQ7 (the largest effect, and the one most worth confirming), on 100 new trials each with seeds that don't overlap the originals (`experiments/rq_replication.py`). The criterion was written down before the rerun: an effect counts as replicated only if the new win rate falls on the same side of 50% as the original and a two-sided sign test on the new trials (ties dropped) gives p < 0.05. The original 30-trial verdicts are left exactly as they were pre-registered; the replication is reported alongside them, not in place of them.

## 3. Results

### 3.1 RQ1: Cluster overlap and the confidently-wrong fit

I generated 600 points from two Gaussians (300 each) at varying separation and fit $K=2$ with EM, predicting that entropy in the ambiguous middle region would rise toward its maximum as the two generating clusters moved closer together. At $c=1.0$ (separation of one standard deviation) it did the opposite: the fitted model's entropy in that region was `0.000`, total confidence, when the Bayes-optimal entropy using the true generating centers was `0.684`. This happened because EM found a lopsided solution, roughly 97% of the data in one component and 3% in the other, that beat the honest 50/50 split on log-likelihood, exactly the unbounded-likelihood mechanism described in Section 2.1. It happened on 9 of 10 random restarts, not once by chance.

### 3.2 RQ2: Is this common, and does a smarter initialization fix it?

A 140-trial sweep found the RQ1 pattern on 14 to 19 of every 30 trials at $c \in \{0.5, 1.0\}$ (14, 19, 15, and 17 across the four overlapping conditions), and on 0 of 20 trials at a well-separated control ($c=4$), confirming it is an overlap-driven phenomenon and not a general bug, but common rather than universal. The more interesting result: switching from random initialization to k-means++ did not reduce the pattern's frequency, if anything it was slightly more common. A plausible reason, which I haven't verified directly: k-means++ is designed to seek out points that sit apart from the rest of the data, and the lopsided solution's small component is, by construction, exactly that kind of outlying clump. A separate, sharper follow-up predicted k-means++'s second starting point would land directly inside the eventual small component about 80% of the time; the real figure was 19.4% (7/36), still about 8 times the 2.4% chance baseline, but a much weaker effect than the first guess.

### 3.3 RQ3: Does uncertainty sampling beat random sampling?

This is the project's central comparison, and the honest summary is that uncertainty sampling showed no advantage over random sampling here. In the original 30 trials at $c=1.0$, entropy-based querying beat random querying in 13 (43.3%), below my pre-committed "this hypothesis is wrong" line of 50%, but that result can't be told apart from a coin flip (p = 0.59). On 100 fresh trials it won 43 (sign test p = 0.19), with an average accuracy gap of 4.7 points in random's favor. The direction is consistent across both runs and my prediction that uncertainty sampling would win 60 to 70% of the time is clearly wrong, but I can't claim it reliably loses. What I traced is why it fails to win: of 600 points in one trial, only 3 sat within 0.001 of maximum entropy, another 30 were moderately ambiguous, and 544 were already confidently assigned. A 60-query budget spent on "always ask about the most uncertain point" exhausts that thin, roughly 55-point ambiguous layer quickly and then keeps circling back to near-duplicates of it, never reaching the 544-point bulk, some of which is confidently *wrong* in a lopsided fit. Random querying has no such redundancy problem; its queries land all over the dataset by chance.

### 3.4 RQ4: Does batching make it worse?

If sequential querying gets stuck revisiting a redundant neighborhood, grabbing the top-10 most uncertain points all at once from one stale ranking should be worse still, since none of the 10 benefit from seeing what the others already revealed. Sequential beat one-shot top-$k$ batching in 20/30 original trials (66.7%), above my predicted range, but at that sample size p = 0.10. On 100 fresh trials sequential won 56 and lost 37, with 7 ties (sign test p = 0.06): the same direction, but short of the bar I set, so this result stays suggestive rather than established. The reasoning still holds in principle (refreshing the ranking after every reveal avoids picking the same redundant point twice, which batching from one snapshot can't do), but in this setting the cost of batching turned out to be small.

### 3.5 RQ5: Four ways uncertainty can fail

**(a) Overcorrection.** A deliberately unrealistic "oracle" strategy that cheats by checking the true label and always queries the model's most *confidently wrong* point should, by leverage alone, beat ordinary entropy sampling. It lost, winning only 12/30 trials (40.0%). That alone wasn't significant (p = 0.36), but the 100-trial replication confirmed it: 35 wins and 64 losses (sign test p = 0.005). Tracing the mixing weight across one trial's queries (0.034 → 0.019 → 0.355 → 0.550 → 0.864, in just a few steps) showed why: clamping is a full-strength, all-or-nothing correction, and a maximally wrong, maximally confident point produces a maximally large correction that can overshoot past the right answer and destabilize the fit, burning later queries on self-inflicted repair instead of new progress.

**(b) True class imbalance.** Keeping the model's belief aside, I tested a truly imbalanced data-generating process (90/10) and measured with balanced accuracy to avoid the trivial trap where a model that only ever guesses the majority class scores 90% while learning nothing. Uncertainty sampling beat random in only 2/30 trials (6.7%, p < 0.001), far below even the pessimistic end of the prediction. Real imbalance gives the RQ1 collapse pathology something genuine to exaggerate, making true minority points look confidently (and wrongly) like majority points, and entropy sampling has no way to notice what it isn't confused about.

**(c) Outliers disguised as boundary points.** Because the two generating clusters in this setup differ only along one axis, entropy depends only on that axis; I placed decoy points at the ambiguous x-coordinate but with an extreme, essentially impossible y-coordinate, so they look exactly as uncertain as a real boundary point while actually being rare and uninformative. Uncertainty sampling beat random in 6/30 trials (20.0%, p = 0.001), confirmed by a direct count: it queried the decoys at 4.5 times the rate chance alone would produce, while random sampling matched chance almost exactly.

**(d) Model misspecification.** Fitting $K=2$ to data generated from 3 true clusters (two of which are structurally forced to merge into one component) sets a hard accuracy ceiling that no querying strategy can get past; worked out by hand before running anything, the ceiling is 66.7%, and the achieved accuracy landed at 66.8%. Here uncertainty sampling won 30/30 trials (100%, p < 0.001, with a caveat below), the opposite direction from every other RQ5 result. The reason is structural: only 8 of 900 points are genuinely, fixably miscategorized, and those 8 carry roughly 30 times the entropy of the rest, so entropy sampling finds nearly all of them within its first several queries while random sampling essentially never does within budget. This is the opposite situation from RQ3: a small, finite, non-regenerating target instead of a large, constantly-refilling one, and targeting wins when the target is finite. A second finding surfaced while checking why some trials exceeded the 66.7% ceiling: the held-out accuracy metric used everywhere else quietly breaks once some points are *permanently* unfixable, since removing a structurally-wrong point from the grading pool (by clamping it) inflates the remaining accuracy without correcting anything. Part of the 100% win rate is the genuine mechanism above and part of it is this metric artifact.

### 3.6 RQ6: Ensemble disagreement without cheating

RQ5a's oracle needed the true label, which no real system has. RQ6 asked whether disagreement across an ensemble of ordinary model fits could act as a usable, non-cheating proxy for "this point might be wrong." Disagreement across 5 independent random restarts beat entropy in 17/30 original trials (56.7%), which on its own wasn't distinguishable from chance (p = 0.58). The replication made it clear: 67 wins and 28 losses on 100 fresh trials (sign test p < 0.001), the strongest replicated effect in the project. It also did better against entropy than RQ5a's cheating oracle did, despite having strictly less information. The mechanism is not that restart-disagreement corrects more gently (clamping strength never changes), but that it tends to flag points without needing any single restart to be extremely confident, which means its picks carry less leverage, and less overcorrection risk, than the oracle's always-extreme targets. A direct follow-up checked whether this win also came from escaping RQ3's separate redundancy problem, and found a real but trial-dependent effect, present in a minority of trials, not the main driver.

Disagreement across 5 independent bootstrap resamples (fitting fresh on data resampled with replacement, rather than from different starting points on the same data) looked even better at first: 18/30 trials (60.0%, p = 0.36). I had predicted it would do no better than entropy, reasoning that a small, tight pathological cluster should be fragile under resampling and produce noisy disagreement. At the time I treated the result as a correction to that reasoning and wrote up an explanation: fragility under resampling as a targeted detector for the small clusters behind the RQ1 collapse. The replication undid that. On 100 fresh trials bootstrap disagreement won 52 and lost 44 (sign test p = 0.48). The best reading now is that it offers no reliable advantage over entropy in this setting, which is what I first predicted, and that my "targeted detector" story was explaining noise. Of everything in the project, this is the clearest lesson in why 30 trials wasn't enough.

### 3.7 RQ7: A real posterior over the parameters

Every result above treats the model's parameters as single best-guess numbers. RQ7 asked what changes if the parameters themselves carry a distribution. I scoped this deliberately to the two cluster means only, holding the mixing weights and covariances fixed at their EM point estimates, which keeps an exact closed-form posterior available as ground truth for validation. I set up and cross-checked two independent ways of approximating the fuller posterior (where the cluster assignments are also uncertain, not held fixed): a Gibbs sampler, alternating between sampling each point's hard cluster assignment and sampling the means from their exact conjugate posterior, and a mean-field variational approximation (CAVI), alternating between computing soft responsibilities and the single best-fitting Gaussian belief over the means, maximizing the same ELBO quantity that underlies ordinary EM's own convergence guarantee.

The two agreed closely on the posterior mean (within 0.015 at a setting with real cluster-assignment ambiguity) and disagreed on the width in exactly the direction theory predicts: the variational approximation came out narrower (80–90% of the sampler's standard deviation), the textbook consequence of forcing an independent approximation onto two variables whose true posterior is entangled.

Built on top of the validated variational machinery, I tested a new strategy, posterior-disagreement: at each query step, draw samples directly from the fitted posterior over the means and query whichever point's predicted cluster varies most across those samples, the principled analogue of RQ6's ad hoc ensembles. This strategy is a close relative of BALD (Houlsby et al., 2011), which I come back to in Section 5. I predicted a similar win rate to RQ6 (50–65%). The first 30 trials gave 24 wins (80.0%, p = 0.001), above my own pre-committed ceiling, and I wrote an explanation for why it beat RQ6's ensembles so clearly: scoping the uncertainty to the means might strip out incidental noise that full-parameter ensembles carry. The replication cut the effect down to size. On 100 fresh trials posterior disagreement won 59 and lost 38 (sign test p = 0.04). The effect is real, but at 59% it sits inside the range I originally predicted, and below restart disagreement's 67% in the same replication. The 80% was most likely a small-sample overestimate, so the "cleaner signal" explanation is no longer needed and I've dropped it. Alongside this, picking a validation seed surfaced a finding worth recording on its own: at the project's usual $c=1.0$, I scanned 20 data seeds and found every single one converged to an imbalanced, near-chance-accuracy initial fit. This had been known to be possible since RQ1; it had not previously been checked across many seeds in one place and found to be the norm.

## 4. Synthesis

A few threads run across all seven research questions:

**Uncertainty is not the same thing as informativeness.** The project's starting hypothesis is false as a blanket statement. Across 130 trials uncertainty sampling never reliably beat random in the standard overlapping setting (RQ3). It lost reliably when the uncertainty came from true class imbalance or from disguised outliers (RQ5b, RQ5c), and won decisively when the useful information was a small, finite set of fixable points (RQ5d). Same signal, same algorithm, and the outcome depends on what is producing the uncertainty.

**Leverage and stability are not the same thing.** A point that would teach the model the most if corrected perfectly (RQ5a's oracle targets) is not automatically a safe point to correct, because the correction itself (full-strength clamping) can overshoot. This one replicated cleanly: the oracle lost to plain entropy in 64 of 100 fresh trials.

**A model's own confident mistakes are a real, reproducible phenomenon, not an edge case.** RQ1's collapse pathology, driven by a basic structural fact about full-covariance Gaussian mixtures, shows up again in RQ5b (amplified by true imbalance), RQ5c (exploited to disguise an outlier), and the seed scan in RQ7 (found to be the default outcome at this project's standard separation, not an occasional accident).

**Disagreement helps, but not every kind of ensemble.** Restart disagreement (67 of 100 fresh trials) and posterior disagreement (59 of 100) both beat plain entropy reliably; bootstrap disagreement (52 of 100) did not. One possible reason, which I haven't tested: restarts and posterior samples vary the model while every fit still sees all 600 points, whereas each bootstrap fit leaves out roughly a third of them, which may add noise faster than useful signal at this sample size.

**Thirty trials wasn't enough, and I only found that out by checking.** Two of the explanations I was most pleased with, bootstrap fragility as a targeted signal and RQ7's "cleaner signal", were explanations of effects that shrank or vanished at 100 trials. The lesson I'm taking forward is to work out how many trials a comparison needs before running it, not after.

## 5. How this connects to prior work

None of the project's main findings is new in kind. I worked through most of them before reading the literature on them, so I think of this section as checking my results against what's known, not as a claim of novelty.

**Uncertainty sampling and its failure modes.** Uncertainty sampling goes back to Lewis and Gale (1994), and Settles' (2009) survey of active learning describes its known weaknesses, including a tendency to query outliers and redundant points. Dasgupta and Hsu (2008) show that strategies which keep querying near the current decision boundary build up a sampling bias, and can fail to reach the best answer even with unlimited labels. That's close to what RQ1 and RQ3 show from the inside: the regions where the model is confidently wrong are exactly the ones entropy never asks about. RQ5c's disguised outliers reproduce the motivation for density-weighted strategies, which multiply uncertainty by how representative a point is of the data (Settles and Craven, 2008; Settles, 2009). RQ5b points in the opposite direction from Ertekin et al. (2007), who found active learning can help with class imbalance. Their setting is a margin-based classifier, while in mine the damage runs through a collapse that is specific to unsupervised GMM fits, so the two results don't contradict each other, but the contrast is worth knowing about.

**Batches.** Kirsch, van Amersfoort and Gal (2019) show that taking the top-$b$ points by individual score can pick redundant points and sometimes does worse than random, which is the concern behind RQ4. In my small setting the cost of batching turned out to be modest (Section 3.4).

**Committees and ensembles.** Querying where a committee of models disagrees is query by committee (Seung, Opper and Sompolinsky, 1992), and building the committee from bootstrap resamples is query by bagging (Abe and Mamitsuka, 1998). RQ6 and RQ6c are small instances of those two ideas. The null result for bootstrap disagreement is one data point in a particular setting, not evidence against query by bagging in general.

**Disagreement under a posterior.** BALD (Houlsby et al., 2011) selects the points where samples from the posterior over parameters disagree most about the outcome, measured by mutual information. RQ7's posterior-disagreement score, the variance of a point's responsibility across posterior samples, is a close relative of that idea rather than the same quantity. I arrived at it while working on RQ7 and only connected it to BALD afterwards.

**The model side.** EM is due to Dempster, Laird and Rubin (1977). The unbounded likelihood behind RQ1's collapse is the standard singularity problem for Gaussian mixtures (Bishop, 2006, Section 9.2.1), and mean-field variational inference's tendency to underestimate posterior variance, which RQ7 measured directly against the Gibbs sampler, is also textbook material (Bishop, 2006, Section 10.1.2; see also Blei, Kucukelbir and McAuliffe, 2017). The k-means++ initialization tested in RQ2 is from Arthur and Vassilvitskii (2007).

What I'd claim the project adds is a small, fully controlled setting where each of these mechanisms can be traced end to end, plus an honest record of which effects survived replication and which didn't.

## 6. Limitations

Several choices were made for tractability and should be read as scope, not oversight. All experiments use two-dimensional, two-component synthetic mixtures; whether these mechanisms carry over to higher dimensions, more components, or real data is untested. The original comparisons used 30 trials, which only had the power to detect large effects; the replication covers the six comparisons that were inconclusive or most surprising, while RQ5b, RQ5c and RQ5d were already significant at 30 trials and haven't been rerun. The replication criterion was fixed before the rerun, but the decision to run it, and the choice of 100 trials, came after I'd seen the original results. The held-out accuracy metric used from RQ3 onward has a known failure mode (RQ5d) once some points become permanently unfixable rather than merely unlabeled. RQ7's Bayesian treatment covers only the cluster means, not the mixing weights or covariances. Several mechanism-level explanations (why k-means++ doesn't help in RQ2, why restart disagreement works) are reasoned from the data at hand rather than tested by a dedicated experiment.

## 7. What I'd do next

First, a proper power analysis for every comparison before running it, and a rerun of RQ5b to RQ5d at the same 100-trial scale. Second, a direct look at what restart disagreement actually selects compared with entropy, since it's now the most reliable strategy in the project and I don't have a tested explanation for why. Third, extending RQ7's Bayesian treatment to the mixing weights and covariances, and comparing the variance-based score against BALD's mutual-information score directly. Finally, the metric artifact found in RQ5d means the held-out accuracy convention should be replaced for any future question involving structurally unfixable points.

## 8. How this was done

I used an AI assistant (Claude, through Claude Code) as a tutor and pair programmer throughout. It taught me the math, wrote most of the code to my specifications, ran the experiments, and helped draft this report. I chose what to investigate, committed to every prediction before its experiment ran, worked through the interpretation of each result, and kept to a rule that nothing stays in the project I can't explain. The README has the fuller account.

## 9. Where everything lives

Every claim above is backed by a committed hypothesis, runnable code, and a full write-up in `research_log/`, one file per research question, each following the same format: research question, hypothesis, setup, results, unexpected results, interpretation, limitations, and the next question it raised. Each log that was part of the replication has an added section reporting it. `README.md` has the locked list of research questions and the project's running list of known gaps; `src/gmmlab/` is the from-scratch model implementation; `experiments/` holds the runnable script behind every result in this report, including `rq_replication.py`, whose per-trial results are saved in `data/replication_results.json`.

## References

Abe, N., & Mamitsuka, H. (1998). Query learning strategies using boosting and bagging. In *Proceedings of the 15th International Conference on Machine Learning* (pp. 1–9).

Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. In *Proceedings of the 18th Annual ACM-SIAM Symposium on Discrete Algorithms* (pp. 1027–1035).

Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.

Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational inference: A review for statisticians. *Journal of the American Statistical Association*, 112(518), 859–877.

Dasgupta, S., & Hsu, D. (2008). Hierarchical sampling for active learning. In *Proceedings of the 25th International Conference on Machine Learning*.

Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society: Series B*, 39(1), 1–38.

Ertekin, S., Huang, J., Bottou, L., & Giles, C. L. (2007). Learning on the border: Active learning in imbalanced data classification. In *Proceedings of the 16th ACM Conference on Information and Knowledge Management* (pp. 127–136).

Houlsby, N., Huszár, F., Ghahramani, Z., & Lengyel, M. (2011). Bayesian active learning for classification and preference learning. arXiv:1112.5745.

Kirsch, A., van Amersfoort, J., & Gal, Y. (2019). BatchBALD: Efficient and diverse batch acquisition for deep Bayesian active learning. In *Advances in Neural Information Processing Systems 32*.

Lewis, D. D., & Gale, W. A. (1994). A sequential algorithm for training text classifiers. In *Proceedings of the 17th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval* (pp. 3–12).

Settles, B. (2009). *Active learning literature survey* (Computer Sciences Technical Report 1648). University of Wisconsin–Madison.

Settles, B., & Craven, M. (2008). An analysis of active learning strategies for sequence labeling tasks. In *Proceedings of the 2008 Conference on Empirical Methods in Natural Language Processing* (pp. 1070–1079).

Seung, H. S., Opper, M., & Sompolinsky, H. (1992). Query by committee. In *Proceedings of the 5th Annual Workshop on Computational Learning Theory* (pp. 287–294).
