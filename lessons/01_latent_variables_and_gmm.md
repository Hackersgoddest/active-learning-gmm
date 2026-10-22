# Lesson 1: Latent-variable models and the Gaussian Mixture Model

Read `00_notation.md` first, and keep it open. Math renders in the VS Code
Markdown preview (`Ctrl+Shift+V`). Each important formula is followed by its
NumPy version.

The one idea the whole project stands on:

> We invent a fictional *generative story* for how the data was produced. The
> story contains a variable we never see (a **latent** variable). Learning means
> inverting the story to recover that hidden variable and the story's numbers.

Get this lesson solid and EM, entropy, and active learning are all variations on
it.

--------------------------------------------------------------------------------

## 1.1  What is a latent variable?

A **latent** (hidden) variable is a part of your model that you never measure.
You observe a data point $\mathbf{x}$; you *hypothesise* that behind each
$\mathbf{x}$ sits a hidden label $z$ that explains it.

Example. You have 5,000 adult heights. The histogram has two bumps. A natural
story: each person belongs to one of two subpopulations, $z \in \{1, 2\}$, and
*given* their subpopulation, height is a bell curve around that group's average.
You recorded only the height $\mathbf{x}$, never the group $z$. So $z$ is latent.

Why use a hidden $z$ instead of fitting the wiggly histogram directly?

- **Structure.** The two-bump shape gets *explained*: you recover group means,
  spreads, and proportions, all interpretable.
- **Soft membership.** For a new height of 172 cm you can ask
  $p(z = 1 \mid \mathbf{x})$ and get a graded answer, not a hard cut.
- **It generalises.** "Pick a hidden thing, then generate from it" is the same
  template behind clustering, topic models, PPCA (there the hidden thing was a
  continuous vector), and hidden Markov models. Here it is a discrete label.

--------------------------------------------------------------------------------

## 1.2  The building block: the multivariate Gaussian

A point $\mathbf{x} \in \mathbb{R}^D$ follows a Gaussian with mean
$\boldsymbol{\mu} \in \mathbb{R}^D$ and covariance
$\boldsymbol{\Sigma} \in \mathbb{R}^{D \times D}$ (symmetric, positive definite)
if its density is

$$
\mathcal{N}(\mathbf{x}\mid\boldsymbol{\mu},\boldsymbol{\Sigma})
= (2\pi)^{-D/2}\,\lvert\boldsymbol{\Sigma}\rvert^{-1/2}
\exp\!\left(-\tfrac{1}{2}\,(\mathbf{x}-\boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right).
$$

```python
def gaussian(x, mu, Sigma):
    d = x.shape[0]
    diff = x - mu
    quad = diff @ np.linalg.inv(Sigma) @ diff                     # a scalar
    norm_const = (2 * np.pi) ** (-d / 2) * np.linalg.det(Sigma) ** (-0.5)
    return norm_const * np.exp(-0.5 * quad)
```

Reading: the density at $\mathbf{x}$ is a constant (the two front factors, there
only so the whole thing integrates to 1) times $\exp$ of minus one-half of a
*distance* term.

That distance term is the **squared Mahalanobis distance**:

$$
\Delta^2(\mathbf{x}) = (\mathbf{x}-\boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu}) \;\ge\; 0.
$$

Reading: "how far is $\mathbf{x}$ from the centre $\boldsymbol{\mu}$, measured in
units of this distribution's own spread and correlation." Setting
$\Delta^2(\mathbf{x}) = \rho^2$ for a constant $\rho$ traces an **ellipse** (in 2-D): a
contour of equal density.

Its shape comes from the eigen-decomposition (symmetric $\boldsymbol{\Sigma}$):

$$
\boldsymbol{\Sigma} = \mathbf{U}\,\boldsymbol{\Lambda}\,\mathbf{U}^\top,
\qquad \boldsymbol{\Lambda} = \operatorname{diag}(\lambda_1,\dots,\lambda_D),
\qquad \mathbf{U}^\top\mathbf{U} = \mathbf{I}.
$$

- columns of $\mathbf{U}$ (eigenvectors) are the **axis directions** of the ellipse;
- $\sqrt{\lambda_i}$ are the **axis half-lengths** (the 1-standard-deviation radius
  along axis $i$);
- $\boldsymbol{\Sigma} = \sigma^2\mathbf{I}$ gives circular contours
  ("spherical" / "isotropic"); diagonal $\boldsymbol{\Sigma}$ gives an
  axis-aligned ellipse; full $\boldsymbol{\Sigma}$ gives a tilted ellipse.

Why Gaussians: a *single* Gaussian has closed-form best-fit parameters (the
sample mean and sample covariance), conditioning and marginalising keep things
Gaussian, and it is the least-committal distribution once you fix a mean and a
covariance.

--------------------------------------------------------------------------------

## 1.3  The Gaussian Mixture Model: the generative story

Parameters, bundled as $\boldsymbol{\theta} = \{\pi_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\}_{k=1}^{K}$:

- $\pi_k$ is the **mixing weight** of component $k$, with

$$
\pi_k \ge 0, \qquad \sum_{k=1}^{K}\pi_k = 1
$$

  (a probability distribution over "which component");
- $\boldsymbol{\mu}_k \in \mathbb{R}^D$, the mean of component $k$;
- $\boldsymbol{\Sigma}_k \in \mathbb{R}^{D\times D}$, the covariance of component $k$.

To generate **one** data point $\mathbf{x}_n$:

$$
z_n \sim \operatorname{Categorical}(\pi_1,\dots,\pi_K),
\qquad
\mathbf{x}_n \mid (z_n = k) \sim \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k).
$$

Reading: **Step 1** draw the hidden label $z_n$ ($=k$ with probability $\pi_k$);
**Step 2** draw the point from component $z_n$. Do this independently for
$n = 1,\dots,N$. The labels $z_n$ are thrown away.

**Joint** probability of "point $n$, and its label is $k$":

$$
p(\mathbf{x}_n, z_n = k)
= p(z_n = k)\,p(\mathbf{x}_n \mid z_n = k)
= \pi_k\,\mathcal{N}(\mathbf{x}_n\mid\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k).
$$

Reading: probability of the label ($\pi_k$) times the density of the point under
that component. Just "Step 1 probability $\times$ Step 2 probability".

**Marginal** density of the point alone: add the joint over every value the
latent could take. This operation is called *marginalising out* $z$:

$$
p(\mathbf{x}_n)
= \sum_{k=1}^{K} p(\mathbf{x}_n, z_n = k)
= \sum_{k=1}^{K} \pi_k\,\mathcal{N}(\mathbf{x}_n\mid\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k).
$$

```python
def gmm_density(x, pi, mu, Sigma):               # pi:(K,) mu:(K,D) Sigma:(K,D,D)
    return sum(pi[k] * gaussian(x, mu[k], Sigma[k]) for k in range(len(pi)))
```

Reading: the density of a point is a **weighted sum of $K$ bell curves**, one per
component, bump $k$ scaled by how common that component is ($\pi_k$).

That it is a **sum**, not a product, is the entire reason mixtures are harder to
fit than a single Gaussian. See 1.6.

--------------------------------------------------------------------------------

## 1.4  Responsibilities: inverting the story

You have fitted $\boldsymbol{\theta}$ and you see a point $\mathbf{x}_n$. What is
the probability it was produced by component $k$? **Bayes' rule** (posterior =
prior $\times$ likelihood, divided by the total):

$$
r_{nk} \;\equiv\; p(z_n = k \mid \mathbf{x}_n)
= \frac{\pi_k\,\mathcal{N}(\mathbf{x}_n\mid\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)}
       {\displaystyle\sum_{j=1}^{K}\pi_j\,\mathcal{N}(\mathbf{x}_n\mid\boldsymbol{\mu}_j,\boldsymbol{\Sigma}_j)},
\qquad
r_{nk} \ge 0, \quad \sum_{k=1}^{K} r_{nk} = 1.
$$

```python
def responsibilities(x, pi, mu, Sigma):
    w = np.array([pi[k] * gaussian(x, mu[k], Sigma[k]) for k in range(len(pi))])
    return w / w.sum()                           # (K,) vector, sums to 1
```

$r_{nk}$ is the **responsibility** of component $k$ for point $n$ (Bishop writes
it $\gamma(z_{nk})$).

- The responsibilities of one point form a probability vector across the $K$
  components: a **soft assignment** ("70% component 1, 30% component 2").
- The denominator is exactly $p(\mathbf{x}_n)$ from 1.3, the same normalising sum.
- Real code does this in **log-space** (add $\ln\pi_k$ and the log-Gaussian, then
  a log-sum-exp) so tiny numbers do not underflow to 0. We build that in Stage 3.

**The vector $\mathbf{r}_n = (r_{n1},\dots,r_{nK})$ is what the whole project
pivots on.**

- In Lesson 2, the E-step of EM *is* computing all the $r_{nk}$.
- Later, its **entropy** is our uncertainty score:

$$
\mathbb{H}[z_n \mid \mathbf{x}_n] = -\sum_{k=1}^{K} r_{nk}\,\ln r_{nk},
\qquad 0 \le \mathbb{H}[z_n \mid \mathbf{x}_n] \le \ln K.
$$

```python
r = R[n]
H = -np.sum(r[r > 0] * np.log(r[r > 0]))         # skip 0*log 0
```

Reading: $\mathbb{H} = 0$ when $\mathbf{r}_n$ is one-hot like $(1,0,0)$, meaning the
model is certain which component; $\mathbb{H} = \ln K$ (its maximum) when
$\mathbf{r}_n$ is flat like $(1/K,\dots,1/K)$, meaning the model has no idea.
Uncertainty sampling will mean: "ask for the label of the point with the largest
$\mathbb{H}$".

--------------------------------------------------------------------------------

## 1.5  Two uses of the same fitted model

- **Density estimation**, where you care about $p(\mathbf{x})$: anomaly detection,
  generating samples, comparing likelihoods.
- **Latent inference / clustering**, where you care about
  $p(z_n \mid \mathbf{x}_n)$: which group, and how confidently.

Both drop out of one fit. This project mostly lives in the second.

--------------------------------------------------------------------------------

## 1.6  Why fitting a mixture is hard  (this motivates all of Lesson 2)

Data $\mathbf{x}_1,\dots,\mathbf{x}_N$, assumed independent. The
**log-likelihood** of the parameters (how well $\boldsymbol{\theta}$ explains the
data; bigger is better):

$$
\mathcal{L}(\boldsymbol{\theta})
= \sum_{n=1}^{N} \ln p(\mathbf{x}_n)
= \sum_{n=1}^{N} \ln\!\left( \sum_{k=1}^{K} \pi_k\,\mathcal{N}(\mathbf{x}_n\mid\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k) \right).
$$

Compare two situations.

**If the labels $z_n$ were known** ("complete data"), using
$\mathbb{1}[z_n = k]$ (1 if true, else 0):

$$
\mathcal{L}_{\text{comp}}(\boldsymbol{\theta})
= \sum_{n=1}^{N}\sum_{k=1}^{K} \mathbb{1}[z_n = k]
  \Big( \ln\pi_k + \ln\mathcal{N}(\mathbf{x}_n\mid\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k) \Big).
$$

The points split cleanly by component. Maximising gives closed forms:
$\boldsymbol{\mu}_k$ is the average of the points with $z_n = k$;
$\boldsymbol{\Sigma}_k$ their sample covariance; $\pi_k$ their fraction of all
points. Easy.

**With $z_n$ unknown**, you are stuck with $\ln$ of a **sum**. You cannot move
the $\ln$ inside the sum, so the $K$ component terms never separate. Setting
$\nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta}) = \mathbf{0}$ gives
a tangled system with **no closed-form solution**. And
$\mathcal{L}(\boldsymbol{\theta})$ is **not concave**: several local maxima, and
(with a full $\boldsymbol{\Sigma}_k$) it can run off to $+\infty$ by collapsing
one component onto a single point.

**EM** (Lesson 2) is the standard way out: guess the responsibilities with the
current parameters (the **E-step**, which "fills in" soft labels), then do the easy
closed-form weighted averages as if those soft labels were real (the **M-step**),
and repeat. It never *decreases* $\mathcal{L}(\boldsymbol{\theta})$ (we will
prove that), but that does **not** guarantee it finds the best solution: it can
settle at a local maximum (or, more precisely, a stationary point) of
$\mathcal{L}$. That is exactly why **where you start EM matters (RQ2)** and why
**overlapping components make it hard (RQ1)**.

--------------------------------------------------------------------------------

## 1.7  What "cluster overlap" means, precisely  (RQ1's independent variable)

Informally: two components overlap when a point sitting between them could
plausibly have come from either. We need a **knob** to turn.

Simplest version, used first: set every $\boldsymbol{\Sigma}_k = \sigma^2\mathbf{I}$
(spherical, equal). For two components define the **separation**

$$
c = \frac{\lVert \boldsymbol{\mu}_i - \boldsymbol{\mu}_j \rVert}{\sigma}.
$$

Reading: the distance between the two centres, in standard deviations.

- Large $c$ (say $\ge 4$): the bells barely touch; almost no overlap; even a
  simple classifier gets the labels nearly perfect.
- Small $c$ (say $\le 1$): the bells sit on top of each other; even the *best
  possible* classifier makes many mistakes; in-between points have
  responsibilities close to flat.

(The literature's version of this separation, the exact error floor it implies,
and what it means for entropy are covered in
`RQ1_prep_separation_and_bayes_error.md`, which you read just before RQ1.)

--------------------------------------------------------------------------------

## Your turn

Bullet points are fine. Where a number is asked for, show the arithmetic. I will correct anything
off with a full worked answer and the reasoning, then we start Lesson 2.

### Comprehension

**Q1: Generative story.**
Write the step-by-step recipe to sample *one* point from a $K = 3$ GMM in 2-D
($D = 2$). List every parameter you must fix beforehand, with its shape and any
constraint.

**Q2: Sum vs product.**
Why is
$p(\mathbf{x}_n) = \sum_{k} \pi_k\,\mathcal{N}(\mathbf{x}_n\mid\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)$
and not a *product* of the components? Name the operation on $z$ that produces the
sum, and say where the joint $p(\mathbf{x}_n, z_n = k)$ comes from.

**Q3: Responsibilities + entropy (numbers).**
For $K = 3$, a point $\mathbf{x}$ has unnormalised weights
$\pi_k\,\mathcal{N}(\mathbf{x}\mid\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k) = (0.02,\ 0.02,\ 0.16)$.

  (a) Compute $\mathbf{r} = (r_1, r_2, r_3)$.
  (b) Compute $\mathbb{H} = -\sum_k r_k \ln r_k$ (natural log).
  (c) The maximum for $K = 3$ is $\ln 3 \approx 1.0986$. Would uncertainty
      sampling call this point informative? Why or why not?

**Q4: Why hard.**
In one or two sentences: why can't we get the best-fit GMM parameters by setting
$\nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta}) = \mathbf{0}$ and
solving, the way we can for a single Gaussian?

**Q5: RQ1 prediction, commit now.**
Fix $\boldsymbol{\Sigma}_k = \sigma^2\mathbf{I}$ for all $k$ and shrink the
separation $c = \lVert\boldsymbol{\mu}_i - \boldsymbol{\mu}_j\rVert / \sigma$ from
large toward small. Predict what happens to:

  (a) the responsibilities / entropy of points lying between two components,
  (b) our ability to recover the true labels,
  (c) EM's sensitivity to where it is initialised.

This becomes the written hypothesis in `research_log/RQ1_cluster_overlap.md`.

### Calibration: reply `solid` / `shaky` / `new to me` for each

**Q6.**
  (a) eigen-decomposition of a symmetric matrix
  (b) Bayes' rule
  (c) expectation and variance of a random variable
  (d) Lagrange multipliers for equality-constrained optimisation
  (e) Jensen's inequality / concavity of $\ln$
