# Review: `make_two_gaussians` ownership questions

*Feedback from my AI tutor on my answers to the six ownership questions for `make_two_gaussians`, kept as a record of what I got wrong and how I fixed it.*

Q1 and Q3's core idea are right. Q2, Q4, Q6 each
need a real correction. Q5 gets a full answer below, since you asked for one.

--------------------------------------------------------------------------------

## Q1: correct

The data, the true labels, the true means. That's the whole return value.

--------------------------------------------------------------------------------

## Q2: the "why $c$" reasoning is right; the call chain is wrong

Your reasoning for *why* $c$ specifically, "so we can test how $c$ affects
EM's ability to recover the latent labels", is exactly right, that's RQ1.

But `fit` / `fit_best_of` don't call `make_two_gaussians`, and it doesn't call
them either. They don't know about each other at all. The actual chain, once
we write `experiments/rq1_cluster_overlap.py`, is:

$$
\texttt{make\_two\_gaussians(c, ...)} \;\longrightarrow\; \mathbf{X}
\;\longrightarrow\; \texttt{fit\_best\_of(X, 2, ...)} \;\longrightarrow\; \text{a fitted model}
\;\longrightarrow\; \text{compare to the TRUE }\mathbf{z}\text{ and }\boldsymbol\mu
$$

The **experiment script** is what calls both, in that order: generate data,
then fit it, then grade the fit against the truth `make_two_gaussians` also
handed back. Worth being precise about this: `gmmlab`'s functions are built to
be composed by something else (a script, a test), not to call each other
beyond what's already wired up (e.g. `em.fit` calling `responsibilities` and
`m_step` internally, that's a real call, *inside* the library).

--------------------------------------------------------------------------------

## Q3: right intuition, the equation you couldn't recall

What you described, placing both means on one axis so responsibility
collapses to one number, is exactly the reasoning from the prep note's Section P.3,
and you reconstructed *why* we do it correctly without the formula in front of
you. The equation itself is just $c$'s own definition, rearranged:

$$
c = \frac{\lVert\boldsymbol\mu_1-\boldsymbol\mu_2\rVert}{\sigma}
\quad\Longrightarrow\quad
\boldsymbol\mu_1 = \Big(\frac{c\sigma}{2},\,0\Big), \qquad
\boldsymbol\mu_2 = \Big(-\frac{c\sigma}{2},\,0\Big)
$$

Check it satisfies the definition: $\lVert\boldsymbol\mu_1-\boldsymbol\mu_2\rVert = c\sigma$,
divide by $\sigma$, get $c$ back. Splitting it symmetrically (half on each
side, rather than putting $\boldsymbol\mu_1$ at the origin and
$\boldsymbol\mu_2$ at distance $c\sigma$) is a convenience, not a requirement;
it's what makes "the midpoint" literally the point $(0,0)$.

--------------------------------------------------------------------------------

## Q4: correct, but there are more assumptions worth naming

Same $\sigma$ for both, yes. Also baked in, silently, by the function's
design:

- **Equal weights.** There's no `pi` parameter at all; it's hardcoded 50/50
  by construction (half the points from each component).
- **Spherical, not just "shared."** $\boldsymbol\Sigma=\sigma^2\mathbf{I}$ means
  the spread is the same in *every direction*, not only the same *between* the
  two components: a circle, not a tilted ellipse.
- **Exactly $K=2$.** The function doesn't generalize to three or more
  components; that's a different function if RQ1 ever needs it.
- **The axis is a choice, not a law.** Lying the means along the x-axis is
  just convenient for reading off "distance from the midpoint" as a single
  x-coordinate; nothing in the math requires that specific axis.

--------------------------------------------------------------------------------

## Q5: full answer, since you said you don't have it yet

"How do we know it's right?" doesn't mean "it ran without crashing." It means:
specific, falsifiable checks, each targeting a different way the function
could be subtly wrong. Here's what each of the 7 tests actually rules out:

1. **`test_true_separation_is_exactly_c`**: checks the *returned* $\boldsymbol\mu$
   itself satisfies $\lVert\boldsymbol\mu_1-\boldsymbol\mu_2\rVert/\sigma = c$,
   exactly, for several $c$. This checks the **formula**: did we place the
   means where the equation says to?
2. **`test_empirical_moments_converge_to_the_truth`**: generates 50,000
   points per component and checks the *sampled* data's mean and per-group
   spread land close to the true $\boldsymbol\mu$ and $\sigma$. This is a
   different check from #1: #1 can pass even if `rng.normal(...)` were called
   wrong (e.g. forgetting to multiply by `sigma`); #1 only checks the
   *returned* `mu`, not what was actually sampled around it. #2 catches that.
3. **`test_shapes`**: the three return values have the sizes the docstring
   promises. Cheap, and catches a wrong `axis` or a transposed array instantly
   instead of as a confusing error three functions downstream.
4. **`test_reproducible_with_same_seed`**: same `rng` seed gives *identical*
   data. This matters for science, not just testing: an experiment has to be
   rerunnable to the same numbers, or a "finding" can't be checked by anyone,
   including you a week later.
5. **`test_different_seeds_give_different_data`**, the opposite check: if #4
   passed by accident because `rng` was silently ignored and the function
   always returns the same fixed data no matter what, this would catch it.
6. **`test_raises_on_invalid_inputs`**: `sigma<=0`, `n_per_component<1`, now
   also `c<0` (added below) all raise `ValueError` instead of silently
   returning nonsense.
7. **`test_combined_array_std_is_not_sigma_once_separated`**: locks in the
   mistake from Q6, below, as a permanent test, so it can't quietly come back
   if the function changes later.

No single test proves correctness alone; each one closes off one *specific*
way the function could be wrong. That's what "verified" means in this
project: not vibes, a list of falsifiable claims that all held.

--------------------------------------------------------------------------------

## Q6: found a real, unguarded case; fixed it

"The std is different for the two clusters" restates Q4's assumption rather
than answering "what breaks it", but asking the question surfaced something
real, in two places:

**1. A genuine gap, now fixed.** I checked: `make_two_gaussians(c=-3, ...)`
did **not** raise an error. It silently placed $\boldsymbol\mu_1=(-1.5,0)$,
$\boldsymbol\mu_2=(1.5,0)$: the two means just swap sides, and the *resulting*
separation is $3$, not $-3$. Harmless by accident (swapping which side is
"which" doesn't change the statistics), but $c$ is a distance divided by a
distance; it cannot be negative by definition, the same way `sigma<=0` can't
be a valid spread. The function silently accepted a value outside its own
definition instead of saying so. Added a `c < 0` check (same pattern as the
other two), and a test for it.

**2. The trap I made myself, re-confirmed with the real numbers.** With
$c=4,\ \sigma=1$: `X.std(axis=0)` on the full, **combined** array came out
`[2.233, 0.999]`, not `[1.0, 1.0]`. Computed only on one group's points,
`X[z==0].std(axis=0)` comes out `[0.995, 0.999]`, correctly close to
$\sigma=1$. The combined array mixes two different means, so its overall
spread is inflated by *how far apart the means are*, on top of $\sigma$; this
is the law of total variance, not a bug in the function. It's now a permanent
test (#7 above), so a future reader hits the test's explanation instead of
re-discovering this the way I did.

--------------------------------------------------------------------------------

## Where this leaves us

Q1, Q3's core, Q4's core are solid. Reread Q2's call-chain correction and Q6's
two findings once, then we move to finalizing the RQ1 design.
