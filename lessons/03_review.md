# Review of your Lesson 3 answers

*Feedback from my AI tutor on my answers to the Lesson 3 questions, kept as a record of what I got wrong and how I fixed it.*

Q1, Q2, Q3 are correct (small notes only). Q5 and Q6
are correct but shallow; read the additions. Q4 is answered in full below,
since you asked for it, with a numeric check.

--------------------------------------------------------------------------------

## Q1: fully correct, no fix needed

$$
\mathcal{J}(x,y,\nu) = x+2y - \nu(x^2+y^2-5)
$$

$$
\frac{\partial\mathcal{J}}{\partial x} = 1-2\nu x = 0 \Rightarrow x=\frac{1}{2\nu},
\qquad
\frac{\partial\mathcal{J}}{\partial y} = 2-2\nu y = 0 \Rightarrow y=\frac{1}{\nu}=2x.
$$

Substituting $y=2x$ into the constraint: $x^2+4x^2=5 \Rightarrow x=\pm1$, so
$(x,y)=(1,2)$ or $(-1,-2)$. $f(1,2)=1+4=5$ and $f(-1,-2)=-1-4=-5$, so $(1,2)$ is
the maximum. Every step is right, including correctly discarding the other
candidate.

--------------------------------------------------------------------------------

## Q2: correct, one notation slip

$\pi_1=4/10=0.4$, $\pi_2=6/10=0.6$, correct. You wrote the check as
"$4=6=10$", but you clearly mean $N_1+N_2=4+6=10=N$. Write it as an addition
next time; "$4=6=10$" reads as three things being equal, which isn't what
happened.

--------------------------------------------------------------------------------

## Q3: correct, one term to fix

$$
\frac{d}{dv}\Big[\frac{(x-v)^2}{\sigma^2}\Big], \qquad u=x-v,\ \ \frac{du}{dv}=-1
$$

$$
\frac{d}{dv}\Big[\frac{u^2}{\sigma^2}\Big] = \frac{d}{du}\Big[\frac{u^2}{\sigma^2}\Big]\cdot\frac{du}{dv}
= \frac{2u}{\sigma^2}\cdot(-1) = \frac{-2u}{\sigma^2} = \frac{-2(x-v)}{\sigma^2}.
$$

Your arithmetic is exactly this. One naming fix: differentiating a function of
a function (here, "square, then divide by $\sigma^2$" applied to $u(v)=x-v$) is
the **chain rule**, not the product rule. The product rule differentiates a
*product* of two functions, e.g. $d/dx[f(x)g(x)]$, not used here. You didn't
answer the second half: this checks the identity box in Section 3.3
($\nabla_v[(x-v)^\top\mathbf{A}(x-v)]=-2\mathbf{A}(x-v)$) in the case $D=1$,
$\mathbf{A}=1/\sigma^2$.

--------------------------------------------------------------------------------

## Q4: full answer, since D=1 makes P a plain number, not a matrix

This is the one thing to take away: **when $D=1$, the precision "matrix"
$P=1/\sigma^2$ is just a single positive number.** There is no matrix calculus
here at all: no transpose, no trace, no symmetry to invoke. Those only matter
in the general $D$-dimensional case of Section 3.4. The 1-D check exists precisely so
you can verify the general claim using calculus you already have, without
matrices. Your attempt reached for `tr(P)` and transposes because you were
still thinking in the general case; that's the gap to close.

**(a)** $P$ is a positive real number, so this is the single-variable
derivative of $\ln$, nothing else:

$$
\frac{d}{dP}\big[\ln P\big] = \frac{1}{P}.
$$

**(b)** $v^2 P$ is *linear* in $P$ (v is a fixed constant here), and the
derivative of $cP$ with respect to $P$ is just $c$:

$$
\frac{d}{dP}\big[v^2 P\big] = v^2.
$$

**(c)** Combine exactly as Section 3.4 does for the general case. The objective terms
involving $P_k$ (dropping the responsibility index for a moment) are
$\tfrac12\ln P_k - \tfrac12(x_n-\mu_k)^2 P_k$, weighted by $r_{nk}$ and summed
over $n$:

$$
\frac{d}{dP_k}\Big[\sum_n r_{nk}\Big(\tfrac12\ln P_k - \tfrac12(x_n-\mu_k)^2 P_k\Big)\Big]
= \frac{N_k}{2}\cdot\frac{1}{P_k} - \frac12\sum_n r_{nk}(x_n-\mu_k)^2.
$$

Set to zero and use $P_k=1/\sigma_k^2$:

$$
\frac{N_k}{2}\,\sigma_k^2 = \frac12\sum_n r_{nk}(x_n-\mu_k)^2
\quad\Longrightarrow\quad
\boxed{\sigma_k^2 = \frac{1}{N_k}\sum_n r_{nk}(x_n-\mu_k)^2}
$$

That is exactly the general $\boldsymbol\Sigma_k$ formula from Section 3.4 with the outer
product $(\mathbf{x}_n-\boldsymbol\mu_k)(\mathbf{x}_n-\boldsymbol\mu_k)^\top$
collapsed to the scalar square $(x_n-\mu_k)^2$. I checked this numerically
against a brute-force grid search over $\sigma^2$ (maximizing the log-likelihood
directly, no calculus): closed form gave $\sigma^2=8.41502$, brute force gave
the same to 5 decimal places.

--------------------------------------------------------------------------------

## Q5: correct, but says *that* not *why*; one addition

Right that $\pi_k$ is constrained ($\sum_k\pi_k=1$) and $\boldsymbol\mu_k$,
$\boldsymbol\Sigma_k$ aren't tied together that way. One nuance worth having:
$\boldsymbol\mu_k$ truly has **no constraint**: any vector in $\mathbb{R}^D$ is
valid. $\boldsymbol\Sigma_k$ technically *does* have a constraint (symmetric,
positive definite), but it's an **open-set** constraint, not an equality like
$\pi$'s; as long as the unconstrained optimum lands *inside* that set (not on
its boundary), ordinary calculus finds it with no multiplier needed. When
$N_k$ is small, the optimum can drift toward the boundary of that set (a
singular, non-invertible $\boldsymbol\Sigma_k$), which is exactly the
"singularities" item already sitting in `README.md`'s known gaps, closed later
by the covariance floor.

--------------------------------------------------------------------------------

## Q6: half the question unanswered, one phrase to fix

You answered the first half well. Two things to add:

1. **"Avoid model misspecification" is backwards.** Fitting `full` doesn't
   avoid misspecification; right now `full` *is* correctly specified (it's a
   superset of the true spherical-tied model, so it can represent the truth
   exactly). Misspecification is something we'll *deliberately introduce
   later*, in RQ5, by fitting a restricted type (e.g. `spherical`) against data
   that isn't spherical. Fitting `full` now is what makes that later comparison
   possible and honest, not what avoids it.
2. **The unanswered half:** what would it mean if `full`-covariance EM *failed*
   to recover an approximately spherical $\boldsymbol\Sigma_k$? Since we already
   gave the model maximal freedom, the failure can't be blamed on the model
   being too restrictive. The real suspects become: too little data per
   component (noisy $N_k$-weighted estimate), overlapping components confusing
   which points get assigned to which $k$ (this is exactly **RQ1**), or a bad
   initialization landing EM in the wrong local optimum (exactly **RQ2**). So
   this question is actually a preview of what RQ1/RQ2 are for: a `full`-model
   failure to recover the true structure is a *symptom* whose cause those two
   experiments are designed to diagnose.

--------------------------------------------------------------------------------

## Where this leaves us

Q1–Q3, Q5–Q6 needed only small fixes, not re-answers. If Q4's derivation reads
clearly now, and Q6's two additions make sense, Stage 2 is done; say so and we
move to Stage 3: building `gmmlab`, starting with the Gaussian log-density
function.
