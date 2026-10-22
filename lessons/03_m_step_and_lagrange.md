# Lesson 3: The M-step, Lagrange multipliers and the closed-form updates

Read `00_notation.md` first (new symbols added for this lesson: $N_k$,
$\mathcal{J}$, $\nu$, $\mathbf{P}_k$, $\mathrm{tr}$).

This closes Stage 2. Lesson 2, Section 2.5 showed that once responsibilities $r_{nk}$
are fixed (the E-step), maximizing the ELBO over $\boldsymbol\theta$ reduces to
maximizing the **expected complete-data log-likelihood**:

$$
Q(\boldsymbol\theta) = \sum_{n=1}^{N}\sum_{k=1}^{K} r_{nk}
\Big[\ln\pi_k + \ln\mathcal{N}(\mathbf{x}_n\mid\boldsymbol\mu_k,\boldsymbol\Sigma_k)\Big],
$$

subject only to $\pi_k\ge0,\ \sum_k\pi_k=1$. We derive the three closed-form
maximizers: $\pi_k$, $\boldsymbol\mu_k$, $\boldsymbol\Sigma_k$. Every formula
below was checked numerically (closed-form beats 200/200 random perturbations;
gradient $\approx\mathbf{0}$ at the closed form) before being written here.

--------------------------------------------------------------------------------

## 3.1  Lagrange multipliers: the general recipe

**Problem shape:** maximize (or minimize) $f(\mathbf{x})$ subject to an equality
constraint $g(\mathbf{x})=0$.

**Intuition.** Picture the contour lines of $f$ and the constraint curve
$g=0$. Walking along the constraint, $f$ increases until the constraint curve
becomes *tangent* to a contour line of $f$; past that point, every direction
that stays on the constraint decreases $f$. At a tangent point, the two
gradients point along the same line:

$$
\nabla f(\mathbf{x}^\star) = \nu\,\nabla g(\mathbf{x}^\star)
$$

for some scalar $\nu$ (the **Lagrange multiplier**), not necessarily equal, but
parallel.

**Recipe.** Form the **Lagrangian**
$\mathcal{J}(\mathbf{x},\nu) = f(\mathbf{x}) - \nu\,g(\mathbf{x})$, set
$\nabla_{\mathbf{x}}\mathcal{J}=\mathbf{0}$ (this reproduces the tangency
condition above) and $\nabla_\nu\mathcal{J}=0$ (this reproduces $g(\mathbf{x})=0$,
i.e. the constraint), then solve the resulting system.

**Worked example.** Maximize $f(x,y)=xy$ subject to $g(x,y)=x+y-10=0$.

$$
\mathcal{J}(x,y,\nu) = xy - \nu(x+y-10)
$$

$$
\frac{\partial \mathcal{J}}{\partial x} = y-\nu = 0,
\qquad
\frac{\partial \mathcal{J}}{\partial y} = x-\nu = 0
\quad\Longrightarrow\quad x=y=\nu.
$$

Substitute into the constraint: $x+y=10 \Rightarrow 2\nu=10 \Rightarrow \nu=5$,
so $x=y=5$ (max value $f=25$).

Notice the **two-part pattern**: the stationarity conditions alone only pin down
$x=y=\nu$ (a *shape*, a relationship between the unknowns); they don't give a
number until the constraint is used to solve for $\nu$. This is exactly the
pattern from Lesson 2, Q4: forcing $y_k=\pi_k\mathcal{N}_k(\mathbf{x}_n)/q_n(k)$
to be a constant pinned $q_n(k)$'s *shape* (proportional to $\pi_k\mathcal{N}_k$),
and only the normalization $\sum_k q_n(k)=1$ pinned the exact value. Same
two-step logic, different problem. Watch for it again in Section 3.3.

--------------------------------------------------------------------------------

## 3.2  The $\pi_k$ update

Only one term of $Q(\boldsymbol\theta)$ involves $\pi_k$:

$$
f(\boldsymbol\pi) = \sum_{n=1}^{N}\sum_{k=1}^{K} r_{nk}\ln\pi_k
= \sum_{k=1}^{K} N_k \ln\pi_k,
\qquad N_k \equiv \sum_{n=1}^{N} r_{nk}.
$$

($N_k$ is the **effective number of points** "belonging" to component $k$,
generally not an integer, since responsibilities are soft. Because each row of
$\mathbf{r}_n$ sums to 1, $\sum_k N_k = \sum_k\sum_n r_{nk} = \sum_n 1 = N$.)

Constraint: $g(\boldsymbol\pi) = \sum_k \pi_k - 1 = 0$. Lagrangian:

$$
\mathcal{J}(\boldsymbol\pi,\nu) = \sum_{k=1}^{K} N_k\ln\pi_k - \nu\Big(\sum_{k=1}^{K}\pi_k - 1\Big).
$$

$$
\frac{\partial \mathcal{J}}{\partial \pi_k} = \frac{N_k}{\pi_k} - \nu = 0
\quad\Longrightarrow\quad \pi_k = \frac{N_k}{\nu}.
$$

This only pins the **shape** ($\pi_k \propto N_k$), same as the worked example.
Use the constraint to pin $\nu$:

$$
1 = \sum_{k=1}^{K}\pi_k = \sum_{k=1}^{K}\frac{N_k}{\nu} = \frac{1}{\nu}\sum_{k=1}^{K}N_k = \frac{N}{\nu}
\quad\Longrightarrow\quad \nu = N.
$$

$$
\boxed{\pi_k = \frac{N_k}{N}}
$$

```python
Nk = R.sum(axis=0)          # (K,), R is (N, K) responsibilities
pi = Nk / N
```

--------------------------------------------------------------------------------

## 3.3  The $\boldsymbol\mu_k$ update

$\boldsymbol\mu_k\in\mathbb{R}^D$ has **no constraint** (it's a free vector), so
this is plain calculus, no Lagrange multiplier needed. Only the terms with
$\boldsymbol\mu_k$ matter:

$$
f(\boldsymbol\mu_k) = -\tfrac12\sum_{n=1}^{N} r_{nk}\,(\mathbf{x}_n-\boldsymbol\mu_k)^\top\boldsymbol\Sigma_k^{-1}(\mathbf{x}_n-\boldsymbol\mu_k).
$$

**Gradient identity needed:** for a symmetric matrix $\mathbf{A}$ and vector
variable $\mathbf{v}$,

$$
\nabla_{\mathbf{v}}\big[(\mathbf{x}-\mathbf{v})^\top\mathbf{A}(\mathbf{x}-\mathbf{v})\big] = -2\mathbf{A}(\mathbf{x}-\mathbf{v}).
$$

*Why:* let $\mathbf{u}=\mathbf{x}-\mathbf{v}$, so $d\mathbf{u}/d\mathbf{v}=-\mathbf{I}$.
The standard quadratic-form gradient is
$\nabla_{\mathbf{u}}[\mathbf{u}^\top\mathbf{A}\mathbf{u}] = 2\mathbf{A}\mathbf{u}$
for symmetric $\mathbf{A}$ (1-D sanity check: $D=1$, $A=a$, gives
$d/du[au^2]=2au$, matches). Chain rule:
$\nabla_{\mathbf{v}} = (d\mathbf{u}/d\mathbf{v})^\top\nabla_{\mathbf{u}} = -2\mathbf{A}\mathbf{u} = -2\mathbf{A}(\mathbf{x}-\mathbf{v})$.

Apply it:

$$
\nabla_{\boldsymbol\mu_k} f = -\tfrac12\sum_n r_{nk}\cdot\big(-2\boldsymbol\Sigma_k^{-1}(\mathbf{x}_n-\boldsymbol\mu_k)\big)
= \boldsymbol\Sigma_k^{-1}\sum_n r_{nk}(\mathbf{x}_n-\boldsymbol\mu_k).
$$

Set to $\mathbf{0}$. Since $\boldsymbol\Sigma_k^{-1}$ is invertible, the bracket
alone must vanish:

$$
\sum_n r_{nk}\mathbf{x}_n = \boldsymbol\mu_k\sum_n r_{nk} = \boldsymbol\mu_k N_k
\quad\Longrightarrow\quad
\boxed{\boldsymbol\mu_k = \frac{1}{N_k}\sum_{n=1}^{N} r_{nk}\,\mathbf{x}_n}
$$

That is the **responsibility-weighted average** of the data, exactly the soft version
of "average the points assigned to $k$".

```python
mu = (R.T @ X) / Nk[:, None]     # X:(N,D), R:(N,K) -> mu:(K,D)
```

--------------------------------------------------------------------------------

## 3.4  The $\boldsymbol\Sigma_k$ update

Also unconstrained (any symmetric positive-definite matrix is allowed), but the
calculus is matrix calculus. The clean route: parametrize by the **precision
matrix** $\mathbf{P}_k = \boldsymbol\Sigma_k^{-1}$ instead of $\boldsymbol\Sigma_k$
itself, since $\ln\mathcal{N}$ is simpler in $\mathbf{P}_k$:

$$
\ln\mathcal{N}(\mathbf{x}\mid\boldsymbol\mu_k,\mathbf{P}_k^{-1})
= \tfrac12\ln\det\mathbf{P}_k - \tfrac{D}{2}\ln(2\pi) - \tfrac12(\mathbf{x}-\boldsymbol\mu_k)^\top\mathbf{P}_k(\mathbf{x}-\boldsymbol\mu_k).
$$

**Two identities needed** (symmetric $\mathbf{P}$, fixed vector $\mathbf{v}$;
1-D sanity check alongside each: set $D=1$, $\mathbf{P}=1/\sigma^2$ a scalar):

$$
\nabla_{\mathbf{P}}\big[\ln\det\mathbf{P}\big] = \mathbf{P}^{-1}
\qquad\text{(1-D: } d/dP[\ln P] = 1/P \checkmark\text{)}
$$

$$
\nabla_{\mathbf{P}}\big[\mathbf{v}^\top\mathbf{P}\mathbf{v}\big] = \mathbf{v}\mathbf{v}^\top
\qquad\text{(1-D: } d/dP[v^2 P] = v^2 \checkmark\text{)}
$$

(Full proofs use trace identities, e.g. $\mathbf{v}^\top\mathbf{P}\mathbf{v}=\mathrm{tr}(\mathbf{P}\mathbf{v}\mathbf{v}^\top)$
and $d\,\mathrm{tr}(\mathbf{PB})/d\mathbf{P}=\mathbf{B}^\top$; see the *Matrix
Cookbook* (Petersen & Pedersen), identities for $\nabla\ln|\mathbf{X}|$ and
$\nabla(\mathbf{x}^\top\mathbf{Xx})$, if you want the complete derivation; not
required here.)

Collect the $\mathbf{P}_k$-dependent terms of $Q$ and differentiate:

$$
\nabla_{\mathbf{P}_k}\Big[\sum_n r_{nk}\big(\tfrac12\ln\det\mathbf{P}_k - \tfrac12(\mathbf{x}_n-\boldsymbol\mu_k)^\top\mathbf{P}_k(\mathbf{x}_n-\boldsymbol\mu_k)\big)\Big]
= \tfrac{N_k}{2}\mathbf{P}_k^{-1} - \tfrac12\sum_n r_{nk}(\mathbf{x}_n-\boldsymbol\mu_k)(\mathbf{x}_n-\boldsymbol\mu_k)^\top.
$$

Set to $\mathbf{0}$, and recall $\mathbf{P}_k^{-1}=\boldsymbol\Sigma_k$:

$$
\boxed{\boldsymbol\Sigma_k = \frac{1}{N_k}\sum_{n=1}^{N} r_{nk}\,(\mathbf{x}_n-\boldsymbol\mu_k)(\mathbf{x}_n-\boldsymbol\mu_k)^\top}
$$

That is the responsibility-weighted sample covariance, using the *just-updated*
$\boldsymbol\mu_k$.

```python
diff = X[:, None, :] - mu[None, :, :]                     # (N, K, D)
outer = diff[:, :, :, None] * diff[:, :, None, :]          # (N, K, D, D)
Sigma = (R[:, :, None, None] * outer).sum(0) / Nk[:, None, None]
```

--------------------------------------------------------------------------------

## 3.5  All three together, and the callback to Lesson 1

$$
\pi_k = \frac{N_k}{N}, \qquad
\boldsymbol\mu_k = \frac{1}{N_k}\sum_n r_{nk}\mathbf{x}_n, \qquad
\boldsymbol\Sigma_k = \frac{1}{N_k}\sum_n r_{nk}(\mathbf{x}_n-\boldsymbol\mu_k)(\mathbf{x}_n-\boldsymbol\mu_k)^\top.
$$

Compare with Lesson 1, Section 1.6's complete-data MLE (labels *known*): identical
formulas with the hard indicator $\mathbb{1}[z_n=k]$ replaced by the soft
$r_{nk}$, and $N_k$ replacing "count of points in $k$". EM's M-step is the
known-label MLE, run on soft labels.

--------------------------------------------------------------------------------

## 3.6  Design decision: which covariance model does `gmmlab` fit?

Real libraries (`sklearn.mixture.GaussianMixture`) offer a `covariance_type`:

| type | $\boldsymbol\Sigma_k$ | update |
|---|---|---|
| `full` | any SPD matrix, one per component | Section 3.4, as derived |
| `diag` | diagonal, one per component | take just the diagonal of Section 3.4's result |
| `spherical` | $\sigma_k^2\mathbf{I}$, one scalar per component | $\sigma_k^2 = \frac{1}{DN_k}\mathrm{tr}\big(\sum_n r_{nk}(\mathbf{x}_n-\boldsymbol\mu_k)(\mathbf{x}_n-\boldsymbol\mu_k)^\top\big)$ (average the diagonal of Section 3.4's result) |
| `tied` | one shared $\boldsymbol\Sigma$ for all $k$ | pool Section 3.4's numerator over $k$, divide by $N$ |

**Decision: `gmmlab` implements `full` as the only M-step for now.** Reasons:

- RQ1's synthetic data is generated as spherical-tied
  ($\boldsymbol\Sigma_k=\sigma^2\mathbf{I}$, same for both components), a
  special case of `full`. Fitting the *more general* model and checking whether
  EM recovers the *simpler* true structure is itself a useful validation check
  in Stage 4, and it's the honest way to test recovery (fitting `spherical` when
  the truth is `spherical` would make recovery artificially easy).
- RQ5 (misspecification) needs to *deliberately* mismatch the fitted model
  against the truth; that experiment is more informative if `full` is already
  available as a baseline to mismatch against.
- It matches the standard library convention, which makes the Stage 4 sanity
  check against `sklearn` a direct comparison.

`diag` / `spherical` / `tied` remain one-line changes to the M-step (see the
table) and can be added later, cheaply, if a specific RQ needs the efficiency or
the restriction itself is the object of study.

This closes the "which covariance model to fit" item in `README.md`'s known
gaps.

--------------------------------------------------------------------------------

## Your turn

Plain text is fine for answers (write `pi_k`, `sqrt`, etc. as text).

**Q1: Lagrange, a new instance.**
Maximize $f(x,y)=x+2y$ subject to $g(x,y)=x^2+y^2-5=0$. Form the Lagrangian,
take the two partial derivatives, and solve (there are two candidate solutions;
which one is the maximum?).

**Q2: $\pi_k$ by hand.**
$N=10$ points, $K=2$. Suppose $N_1=\sum_n r_{n1}=4$ and $N_2=\sum_n r_{n2}=6$.
Compute $\pi_1,\pi_2$, and confirm $N_1+N_2=N$.

**Q3: the $\boldsymbol\mu_k$ identity in 1-D.**
Confirm by direct differentiation (no matrices, $D=1$) that
$\frac{d}{dv}\big[(x-v)^2/\sigma^2\big] = -2(x-v)/\sigma^2$, and say which line
of Section 3.3 this checks.

**Q4: the $\boldsymbol\Sigma_k$ identities in 1-D.**
With $D=1$ and precision $P=1/\sigma^2$: (a) confirm $d/dP[\ln P]=1/P$; (b)
confirm $d/dP[v^2 P] = v^2$; (c) combine them the way Section 3.4 does, and check you
land on $\sigma_k^2 = \frac{1}{N_k}\sum_n r_{nk}(x_n-\mu_k)^2$.

**Q5: constrained vs. unconstrained.**
In your own words: why does $\pi_k$ need a Lagrange multiplier while
$\boldsymbol\mu_k$ and $\boldsymbol\Sigma_k$ don't?

**Q6: the covariance decision.**
In your own words: why do we fit `full` covariance for RQ1 even though the data
are generated with `spherical`-`tied` covariance? What would we learn if EM,
given `full` freedom, *failed* to recover a spherical-looking $\boldsymbol\Sigma_k$?
