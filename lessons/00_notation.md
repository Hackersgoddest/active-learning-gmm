# Notation cheat sheet

Keep this open while reading the lessons.

## How the math renders

The lessons use LaTeX math: `$ ... $` inline and `$$ ... $$` for display equations.

- **VS Code**: open the file and press `Ctrl+Shift+V` (Cmd on macOS) for the
  Markdown preview. Math renders there by default (`markdown.math.enabled`).
- **GitHub**: renders `$` / `$$` math automatically in `.md` files.

If a formula shows as raw `\mathcal{N}...` text, you are looking at the plain
source, not the preview.

Every important formula is followed by a `python` block with the NumPy version,
so the code never gets ahead of the math.

## Typography convention (same as the MML book / Murphy PML)

| kind             | style                    | examples                                   |
|------------------|--------------------------|--------------------------------------------|
| scalar           | italic, lower or upper   | $x$, $N$, $K$, $D$, $\pi_k$, $\lambda_i$   |
| vector           | **bold lower**           | $\mathbf{x}$, $\boldsymbol{\mu}_k$         |
| matrix           | **bold upper**           | $\boldsymbol{\Sigma}_k$, $\mathbf{U}$, $\mathbf{I}$ |
| space / operator | blackboard               | $\mathbb{R}^D$ (space); $\mathbb{E}$, $\mathbb{H}$, $\mathbb{1}$ (operators) |
| all params bundle| bold theta               | $\boldsymbol{\theta}$                      |

## Symbol glossary

| symbol | LaTeX source | meaning | NumPy |
|---|---|---|---|
| $\mathbf{x}_n$ | `\mathbf{x}_n` | the $n$-th data point, a vector in $\mathbb{R}^D$ | `X[n]`  (`X` is `(N, D)`) |
| $D$ | `D` | dimension = features per point | `X.shape[1]` |
| $N$ | `N` | number of data points | `X.shape[0]` |
| $K$ | `K` | number of mixture components | `K` |
| $z_n$ | `z_n` | hidden (latent) label of point $n$: which component made it | `z[n]` |
| $\boldsymbol{\mu}_k$ | `\boldsymbol{\mu}_k` | mean vector of component $k$, in $\mathbb{R}^D$ | `mu[k]`  (`mu` is `(K, D)`) |
| $\boldsymbol{\Sigma}_k$ | `\boldsymbol{\Sigma}_k` | covariance matrix of component $k$, $D\times D$, symmetric positive definite | `Sigma[k]`  (`(K, D, D)`) |
| $\pi_k$ | `\pi_k` | mixing weight of component $k$ | `pi[k]`  (`pi` is `(K,)`) |
| $\boldsymbol{\theta}$ | `\boldsymbol{\theta}` | all parameters: $\{\pi_k,\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k\}_{k=1}^K$ | `(pi, mu, Sigma)` |
| $\mathcal{N}(\mathbf{x}\mid\boldsymbol{\mu},\boldsymbol{\Sigma})$ | `\mathcal{N}(\mathbf{x}\mid\boldsymbol{\mu},\boldsymbol{\Sigma})` | Gaussian density at $\mathbf{x}$ | `gaussian(x, mu, Sigma)` |
| $r_{nk}$ | `r_{nk}` | responsibility: $p(z_n=k\mid\mathbf{x}_n)$. Bishop writes $\gamma(z_{nk})$ | `R[n, k]`  (`R` is `(N, K)`) |
| $p(A)$ | `p(A)` | probability / density of $A$ | n/a |
| $p(A\mid B)$ | `p(A\mid B)` | probability of $A$ given $B$ is known | n/a |
| $p(A,B)$ | `p(A,B)` | joint probability of $A$ and $B$ | n/a |

### Added by Lesson 1 (Section 1.7), Lesson 2, and the RQ1 prep note

| symbol | LaTeX source | meaning | NumPy |
|---|---|---|---|
| $\sigma$ | `\sigma` | standard deviation, when $\boldsymbol\Sigma=\sigma^2\mathbf{I}$ | `sigma` |
| $\mathbf{I}$ | `\mathbf{I}` | identity matrix | `np.eye(D)` |
| $c$ | `c` | our separation $\lVert\boldsymbol\mu_i-\boldsymbol\mu_j\rVert/\sigma$ (RQ1's knob) | `c` |
| $c_{\text{D}}$ | `c_{\text{D}}` | Dasgupta's separation; $c=\sqrt{D}\,c_{\text{D}}$ for equal spherical $\boldsymbol\Sigma$ | n/a |
| $\Delta^2(\mathbf{x})$ | `\Delta^2(\mathbf{x})` | squared Mahalanobis distance | see core formulas |
| $\mathbf{U},\ \boldsymbol\Lambda,\ \lambda_i$ | `\mathbf{U}, \boldsymbol\Lambda, \lambda_i` | eigenvectors, diagonal eigenvalue matrix, eigenvalues of $\boldsymbol\Sigma$ | `lam, U = np.linalg.eigh(Sigma)` |
| $\Phi$ | `\Phi` | standard normal CDF | `scipy.stats.norm.cdf` |
| $\operatorname{sigmoid}(u)$ | `\operatorname{sigmoid}(u)` | $1/(1+e^{-u})$ | `1/(1+np.exp(-u))` |
| $\mathcal{L}(\boldsymbol\theta)$ | `\mathcal{L}(\boldsymbol\theta)` | log-likelihood of the data | `loglik` |
| $\mathcal{L}_{\text{comp}}$ | `\mathcal{L}_{\text{comp}}` | complete-data log-likelihood (labels known) | n/a |
| $\nabla_{\boldsymbol\theta}$ | `\nabla_{\boldsymbol\theta}` | gradient with respect to $\boldsymbol\theta$ | n/a |
| $\sim$ | `\sim` | "is drawn from" | `rng.normal(...)` etc. |
| $\operatorname{Categorical}(\pi_1,\dots,\pi_K)$ | `\operatorname{Categorical}` | pick $k$ with probability $\pi_k$ | `rng.choice(K, p=pi)` |
| $\varepsilon$ | `\varepsilon` | small positive number (covariance floor) | `eps` |
| $q_n(k)$ | `q_n(k)` | a distribution over the labels for point $n$ that we *choose* | `Q[n, k]` |
| $\boldsymbol\theta^{(t)}$ | `\boldsymbol\theta^{(t)}` | parameters after $t$ EM iterations | n/a |
| $\mathrm{ELBO}(q,\boldsymbol\theta)$ | `\mathrm{ELBO}(q,\boldsymbol\theta)` | lower bound on $\mathcal{L}(\boldsymbol\theta)$ | n/a |
| $\mathrm{KL}(q\Vert p)$ | `\mathrm{KL}(q\Vert p)` | $\sum_k q(k)\ln\frac{q(k)}{p(k)}\ge0$ | `np.sum(q*np.log(q/p))` |

## Symbol collisions to watch for

- $\pi_k$ (a mixing weight) vs. $\pi$ (the number 3.14159) as in $(2\pi)^{-D/2}$.
  The subscript tells you which.
- $N$ (number of points) vs. $\mathcal{N}(\cdot)$ (Gaussian). Different fonts.
- $c$ is separation. The constant in an ellipse equation is called $\rho^2$, and
  the common value in Lesson 2's equality argument is $C$.
- $\mathcal{L}(\boldsymbol\theta)$ is the log-likelihood, and the bound is
  $\mathrm{ELBO}$. Bishop writes the bound as $\mathcal{L}(q,\boldsymbol\theta)$,
  so expect that when reading his book.
- $D$ is always the dimension. When we later need the dataset we will write
  $\mathcal{D}$.

### Added by Lesson 3

| symbol | LaTeX source | meaning | NumPy |
|---|---|---|---|
| $N_k$ | `N_k` | effective count of component $k$: $\sum_n r_{nk}$. $\sum_k N_k = N$ | `R.sum(0)[k]` |
| $\mathcal{J}(\cdot,\nu)$ | `\mathcal{J}(\cdot,\nu)` | a Lagrangian (objective plus $\nu\times$constraint) | n/a |
| $\nu$ | `\nu` | a Lagrange multiplier | n/a |
| $\mathbf{P}_k$ | `\mathbf{P}_k` | precision matrix of component $k$: $\mathbf{P}_k=\boldsymbol\Sigma_k^{-1}$ | `np.linalg.inv(Sigma[k])` |
| $\mathrm{tr}(\cdot)$ | `\mathrm{tr}(\cdot)` | trace of a matrix (sum of diagonal entries) | `np.trace(...)` |

## Reserved for later lessons

$\mathcal{U}$ unlabeled pool, $\mathcal{D}_\ell$ labeled set, $B$ label budget,
$a(\mathbf{x})$ acquisition score, $y_n$ revealed label. Introduced when we reach
RQ3.

## Operators

| symbol | LaTeX source | meaning | NumPy |
|---|---|---|---|
| $\displaystyle\sum_{k=1}^{K} a_k$ | `\sum_{k=1}^{K} a_k` | $a_1 + a_2 + \dots + a_K$ | `a.sum()` / `sum(a[k] for k in range(K))` |
| $\displaystyle\prod_{n=1}^{N} a_n$ | `\prod_{n=1}^{N} a_n` | $a_1 a_2 \cdots a_N$ | `a.prod()` |
| $\mathbf{x}^\top$ | `\mathbf{x}^\top` | transpose | `x.T` |
| $\boldsymbol{\Sigma}^{-1}$ | `\boldsymbol{\Sigma}^{-1}` | matrix inverse | `np.linalg.inv(Sigma)` |
| $\lvert\boldsymbol{\Sigma}\rvert$ or $\det\boldsymbol{\Sigma}$ | `\det\boldsymbol{\Sigma}` | determinant (one number) | `np.linalg.det(Sigma)` |
| $\mathbf{u}^\top\mathbf{v}$ | `\mathbf{u}^\top\mathbf{v}` | dot product | `u @ v` |
| $\boldsymbol{\Sigma}\,\mathbf{x}$ | `\boldsymbol{\Sigma}\,\mathbf{x}` | matrix times vector | `Sigma @ x` |
| $\lVert\mathbf{v}\rVert$ | `\lVert\mathbf{v}\rVert` | Euclidean length $\sqrt{\mathbf{v}^\top\mathbf{v}}$ | `np.linalg.norm(v)` |
| $\mathbb{E}_Y[f(Y)]$ | `\mathbb{E}_Y[f(Y)]` | expectation: weighted average $\sum_k w_k f(y_k)$, weights $w_k$ = probabilities | `np.sum(w * f(y))` |
| $\mathbb{H}[z]$ | `\mathbb{H}[z]` | entropy of a discrete distribution $p$: $-\sum_k p_k\ln p_k$ | `-np.sum(p[p>0] * np.log(p[p>0]))` |
| $\mathbb{1}[\,\cdot\,]$ | `\mathbb{1}[\cdot]` | 1 if the condition holds, else 0 | `(cond).astype(float)` |
| $\ln$ | `\ln` | natural log (base $e$); "log" always means this | `np.log` |

## The core formulas of Lesson 1

### Gaussian density

$$
\mathcal{N}(\mathbf{x}\mid\boldsymbol{\mu},\boldsymbol{\Sigma})
= (2\pi)^{-D/2}\,\lvert\boldsymbol{\Sigma}\rvert^{-1/2}
\exp\!\left(-\tfrac{1}{2}\,(\mathbf{x}-\boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)
$$

```python
def gaussian(x, mu, Sigma):
    d = x.shape[0]
    diff = x - mu
    quad = diff @ np.linalg.inv(Sigma) @ diff                     # scalar
    norm_const = (2 * np.pi) ** (-d / 2) * np.linalg.det(Sigma) ** (-0.5)
    return norm_const * np.exp(-0.5 * quad)
```

### Squared Mahalanobis distance

$$
\Delta^2(\mathbf{x}) = (\mathbf{x}-\boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})
$$

```python
delta2 = (x - mu) @ np.linalg.inv(Sigma) @ (x - mu)
```

### Mixing-weight constraints

$$
\pi_k \ge 0, \qquad \sum_{k=1}^{K}\pi_k = 1
$$

### GMM density of one point

$$
p(\mathbf{x}_n) = \sum_{k=1}^{K} \pi_k\,\mathcal{N}(\mathbf{x}_n\mid\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)
$$

```python
def gmm_density(x, pi, mu, Sigma):                # pi:(K,) mu:(K,D) Sigma:(K,D,D)
    return sum(pi[k] * gaussian(x, mu[k], Sigma[k]) for k in range(len(pi)))
```

### Responsibility of component $k$ for point $n$

$$
r_{nk} \equiv p(z_n = k \mid \mathbf{x}_n)
= \frac{\pi_k\,\mathcal{N}(\mathbf{x}_n\mid\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)}
       {\displaystyle\sum_{j=1}^{K}\pi_j\,\mathcal{N}(\mathbf{x}_n\mid\boldsymbol{\mu}_j,\boldsymbol{\Sigma}_j)}
\qquad\text{with}\qquad r_{nk}\ge 0,\quad \sum_{k=1}^{K} r_{nk} = 1
$$

```python
def responsibilities(x, pi, mu, Sigma):
    w = np.array([pi[k] * gaussian(x, mu[k], Sigma[k]) for k in range(len(pi))])
    return w / w.sum()                            # (K,) vector, sums to 1
```

### Assignment entropy (the uncertainty score)

$$
\mathbb{H}[z_n \mid \mathbf{x}_n] = -\sum_{k=1}^{K} r_{nk}\,\ln r_{nk},
\qquad 0 \le \mathbb{H}[z_n \mid \mathbf{x}_n] \le \ln K
$$

```python
r = R[n]
H = -np.sum(r[r > 0] * np.log(r[r > 0]))         # skip 0*log 0
```

### Log-likelihood of the parameters

$$
\mathcal{L}(\boldsymbol{\theta})
= \sum_{n=1}^{N} \ln p(\mathbf{x}_n)
= \sum_{n=1}^{N} \ln\!\left( \sum_{k=1}^{K} \pi_k\,\mathcal{N}(\mathbf{x}_n\mid\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k) \right)
$$

```python
loglik = sum(np.log(gmm_density(X[n], pi, mu, Sigma)) for n in range(len(X)))
```
