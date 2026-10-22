# Stage 3, function 1: `gaussian_logpdf`, review of your six answers

*Feedback from my AI tutor on my answers to the six ownership questions for `gaussian_logpdf`, kept as a record of what I got wrong, and of the bug in its own code that my answers caught.*

## First, a bug in *my* code that your Q4 and Q6 exposed

My docstring claimed `sign <= 0` from `slogdet` "catches a Sigma that isn't
positive definite". That is false. The sign of $\det\boldsymbol\Sigma$ is the
product of the eigenvalue signs, so a matrix with an **even number** of negative
eigenvalues has $\det>0$ and slips through. I reproduced three silent failures
before fixing them:

| input | what the old code did | correct behaviour |
|---|---|---|
| $\boldsymbol\Sigma=-\mathbf{I}$ (D=2, det $=+1$) | returned a number | raise: not positive definite |
| $\boldsymbol\Sigma=\begin{pmatrix}2&1\\0&2\end{pmatrix}$ (not symmetric) | returned a number | raise: not symmetric |
| `mu` of shape `(1, 2)` | $D$ silently became 1; result off by $0.919$ | raise: `mu` must be `(D,)` |

My original test only tried $\begin{pmatrix}1&2\\2&1\end{pmatrix}$ (det $=-3$), which
the sign check happened to catch. That is a test chosen so my code passes, not
one designed to break it. Fixed: explicit shape checks, a symmetry check, and
`eigvalsh(Sigma).min() > 0`. Tests went from 6 to 10 and now include each case
above. **Lesson for the project:** a passing test only means something if it was
designed to fail a wrong implementation.

--------------------------------------------------------------------------------

## Your answers

**Q1: correct.** Add the shape: one component, every row of `X`, returns `(N,)`.

**Q2: correct.** Responsibilities and the log-likelihood are exactly its two
callers.

**Q3: circular; this is the one to redo.** "The log of the Gaussian density,
which comes from taking the log of the Gaussian density" doesn't say either the
equation or *why* we take the log. The equation:

$$
\ln\mathcal{N}(\mathbf{x}\mid\boldsymbol\mu,\boldsymbol\Sigma)
= -\tfrac12\Big[\,D\ln(2\pi) + \ln\lvert\boldsymbol\Sigma\rvert
+ (\mathbf{x}-\boldsymbol\mu)^\top\boldsymbol\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)\Big].
$$

Every piece maps to a line of code: `D * np.log(2*np.pi)`, `logdet` from
`slogdet`, and `quad` from the `solve` + `einsum`. **Why the log:** at the point
$(40,40)$ with $\boldsymbol\mu=\mathbf{0}$, $\boldsymbol\Sigma=\mathbf{I}$, the raw
density is $\tfrac{1}{2\pi}e^{-1600}$, which float64 rounds to exactly `0.0`. The
log-density is $-1601.84$, a perfectly ordinary number. There is now a test for
this (`test_finite_where_raw_density_underflows`). Later, a product of many
densities becomes a sum of logs, which is what the log-likelihood needs.

**Q4: right, incomplete.** Symmetric positive definite, yes. Also the shapes:
`X` is `(N, D)`, `mu` is `(D,)`, `Sigma` is `(D, D)`. And note the difference
between *assuming* and *checking*: the old code assumed symmetry without
checking it.

**Q5: correct and specific.** Each test maps to a claim. The honest addition is
what the tests did *not* cover, which is how the bug above survived: no
negative-definite input, no asymmetric input, no shape misuse, no underflow case.

**Q6: right instinct, but not the dangerous case.** Your example (`X` with 3
columns against a length-2 `mu`) makes NumPy raise a broadcasting error on its
own, so it fails loudly. The dangerous failures are the **silent** ones in the
table above. When asking "what breaks it?", separate *fails loudly* from *returns
a wrong number*; the second kind is what costs weeks in research code.

--------------------------------------------------------------------------------

## Before we move on

Redo Q3 in your own words: write the equation (plain text is fine) and say why
the log matters, using the $(40,40)$ example. Then log-sum-exp is next.
