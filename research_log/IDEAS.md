# Parking lot

Ideas for later RQs. Writing them here means we don't have to design them now.
Nothing here is a commitment. Add a line whenever a thought would otherwise
derail the current stage.

| idea | belongs to |
|---|---|
| Responsibilities are normalized, so a far-away outlier can look "confident". Compare entropy with log-density $\ln p(\mathbf{x})$. | RQ5 |
| Top-k entropy may pick k near-identical points at one boundary. | RQ4 |
| Entropy at the midpoint is $\ln 2$ for every separation; what changes is the mass near the middle. | RQ1 (see prep note) |
| Restart-ensemble disagreement = optimization uncertainty; bootstrap disagreement = data-sampling uncertainty. Keep separate. | RQ6 |
| Posterior over parameters via Gibbs or variational inference. | RQ7 |
| Uncertainty sampling may lose to random because boundary points are unrepresentative. Write down what a loss would mean before running. | RQ3 |
| **Reproducible example, found while testing the EM loop (Stage 3):** two well-separated true clusters (mu = (-6,0) and (6,0), unit variance), fit with `gmmlab.em.fit`. Seed 4 (`np.random.default_rng(4)` for both data and init) converges monotonically (the Section 2.6 guarantee holds) to means near (0,0) instead of the true clusters, since both of EM's K=2 initial means happened to land in the *same* true cluster. Surveyed 30 seeds: 23/30 recovered the truth, 7/30 didn't, and every failure had same-side initial means. See `tests/test_em.py::test_most_seeds_recover_the_true_structure_but_not_all`. A ready-made starting point for RQ2 rather than a constructed example. | RQ2 |
| **Second reproducible example, found while adding the Stage 4 harder-case comparison:** 3 *overlapping*, differently-shaped (non-spherical) true clusters, fit with `gmmlab.em.fit`. One specific seed (`rng=np.random.default_rng(2)`, data via `np.random.default_rng(0)`) converges, again monotonically, to a log-likelihood of -1717.35, about 96.7 points worse than the best of 10 restarts (-1620.70, which matches sklearn's -1620.75 and recovers the true means). Unlike the first example, this is on overlapping (not well-separated) data and exercises full (not spherical) covariance recovery, a harder, more realistic regime for RQ2 to start from. See `tests/test_em.py::test_single_restart_can_land_in_a_worse_local_optimum`. | RQ2 |
| **A degenerate component is maximally "confident," not uncertain.** Found while finalizing RQ1 (README known gaps, "Singularities"): a near-singular component collapsed onto 3 of 600 points at $c=2$, eigenvalue $2\times10^{-5}$. A near-singular Gaussian assigns extremely concentrated, narrow density, so responsibilities for its own points would be near $1.0$ (entropy near $0$), the model's most "certain" outputs in the whole fit, while actually being the single least trustworthy part of it. A direct, concrete instance of "uncertainty ≠ informativeness" (the project's own central caveat), not a hypothetical. `gmmlab.em.is_degenerate` already exists as the flag; RQ5 could use it directly as a misspecification/failure condition to construct, rather than one to merely avoid. | RQ5 |
