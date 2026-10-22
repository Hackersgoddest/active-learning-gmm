# Lesson 2: Jensen's inequality and the ELBO (why EM is even allowed to work)

Read `00_notation.md` first. New symbols used here:

| symbol | meaning |
|---|---|
| $q_n(k)$ | any distribution we *choose* to guess with, over the $K$ labels, for point $n$ ($q_n(k)\ge0$, $\sum_k q_n(k)=1$) |
| $\theta^{(t)}$ | the parameters after $t$ EM iterations |
| $\mathbb{E}_{Y}[f(Y)]$ | expectation: if $Y=y_k$ with probability $w_k$, then $\mathbb{E}_Y[f(Y)] = \sum_k w_k f(y_k)$, a *weighted average* |
| $\mathrm{ELBO}$ | "Evidence Lower BOund", a quantity we build in Section 2.3 that always sits at or below $\mathcal{L}(\boldsymbol\theta)$ |

Lesson 1 left us stuck: $\mathcal{L}(\boldsymbol\theta) = \sum_n \ln\!\big(\sum_k \pi_k\mathcal{N}(\mathbf{x}_n\mid\boldsymbol\mu_k,\boldsymbol\Sigma_k)\big)$
has a $\ln$ wrapped around a sum, so we can't solve $\nabla_{\boldsymbol\theta}\mathcal{L}=\mathbf{0}$
directly. This lesson derives the trick that gets around it, and *proves*, not just
asserts, that the trick (EM) can never make things worse.

--------------------------------------------------------------------------------

## 2.1  Concave functions

A function $f$ is **concave** on an interval if, for any two points $y_1, y_2$ in
that interval and any mixing weight $t \in [0,1]$:

$$
f\big(t\,y_1 + (1-t)\,y_2\big) \;\ge\; t\,f(y_1) + (1-t)\,f(y_2).
$$

Reading: take any two points on the graph of $f$, draw the straight line (chord)
between them; the *curve itself* lies **on or above** that chord, everywhere
between them. ($f(x)=x^2$ is the opposite case, *convex*, with the chord above the curve.)

**Test:** if $f$ is twice differentiable, $f$ is concave wherever $f''(y) \le 0$.

**$\ln$ is concave on $(0,\infty)$:**

$$
\frac{d}{dy}\ln y = \frac{1}{y}, \qquad \frac{d^2}{dy^2}\ln y = -\frac{1}{y^2} < 0 \ \ \text{for all } y>0.
$$

So $\ln$ curves downward everywhere it's defined for us, exactly the shape we'll
exploit.

--------------------------------------------------------------------------------

## 2.2  Jensen's inequality

The two-point definition above extends to *any finite weighted average*, which is
exactly an expectation over a discrete random variable. Let $w_1,\dots,w_K \ge 0$
with $\sum_k w_k = 1$ (a valid probability distribution), and let $y_1,\dots,y_K$
be any numbers. Define a random variable $Y$ that equals $y_k$ with probability
$w_k$; then $\mathbb{E}_Y[Y] = \sum_k w_k y_k$ is exactly the weighted average of
the $y_k$. **Jensen's inequality**, for $f$ concave:

$$
f\Big(\sum_{k=1}^{K} w_k\,y_k\Big) \;\ge\; \sum_{k=1}^{K} w_k\,f(y_k)
\qquad\Longleftrightarrow\qquad
f\big(\mathbb{E}_Y[Y]\big) \;\ge\; \mathbb{E}_Y\big[f(Y)\big].
$$

Reading: **"$f$ of the average" is at least as big as "the average of $f$."**
This follows from the two-point definition by induction on $K$ (each extra term
just splits one more chord); we won't re-derive the induction, we'll use the
result.

**Equality** holds exactly when $f$ is linear on the relevant range, or (the
case we need) when every $y_k$ with $w_k > 0$ takes the **same value**. (If all
the $y_k$ are equal to some constant $c$, both sides are just $f(c)$.)

**Worked numeric check.** $f=\ln$, $y=(1,4)$, $w=(0.5,0.5)$:

$$
\ln(0.5\cdot1 + 0.5\cdot4) = \ln(2.5) \approx 0.916,
\qquad
0.5\ln1 + 0.5\ln4 = 0 + 0.5(1.386) \approx 0.693.
$$

$0.916 \ge 0.693$, confirmed, with room to spare because $1 \ne 4$ (not equal,
so the inequality is strict, not equality).

```python
import numpy as np
lhs = np.log(0.5*1 + 0.5*4)
rhs = 0.5*np.log(1) + 0.5*np.log(4)
lhs, rhs, lhs >= rhs           # (0.9162907..., 0.6931471..., True)
```

--------------------------------------------------------------------------------

## 2.3  Building a lower bound on the log-likelihood

Take one data point $\mathbf{x}_n$. Its log-density is the troublesome
$\ln$-of-a-sum:

$$
\ln p(\mathbf{x}_n) = \ln\sum_{k=1}^{K} \pi_k\,\mathcal{N}(\mathbf{x}_n\mid\boldsymbol\mu_k,\boldsymbol\Sigma_k).
$$

**The trick:** introduce *any* distribution over the labels,
$q_n(k) \ge 0$, $\sum_k q_n(k) = 1$; think of it as "a guess at
$p(z_n=k\mid\mathbf{x}_n)$ that we get to pick." Multiply and divide each term by
$q_n(k)$ (legal as long as $q_n(k) > 0$ wherever we evaluate it):

$$
\ln\sum_{k=1}^{K} \pi_k\,\mathcal{N}(\mathbf{x}_n\mid\boldsymbol\mu_k,\boldsymbol\Sigma_k)
= \ln\sum_{k=1}^{K} q_n(k)\cdot\frac{\pi_k\,\mathcal{N}(\mathbf{x}_n\mid\boldsymbol\mu_k,\boldsymbol\Sigma_k)}{q_n(k)}.
$$

The right side is now exactly $\ln\big(\sum_k w_k\,y_k\big)$ with
$w_k = q_n(k)$ and $y_k = \pi_k\mathcal{N}(\mathbf{x}_n\mid\boldsymbol\mu_k,\boldsymbol\Sigma_k)/q_n(k)$.
Apply Jensen ($\ln$ is concave):

$$
\ln\sum_{k=1}^{K} q_n(k)\cdot\frac{\pi_k\,\mathcal{N}(\mathbf{x}_n\mid\boldsymbol\mu_k,\boldsymbol\Sigma_k)}{q_n(k)}
\;\ge\;
\sum_{k=1}^{K} q_n(k)\,\ln\!\left(\frac{\pi_k\,\mathcal{N}(\mathbf{x}_n\mid\boldsymbol\mu_k,\boldsymbol\Sigma_k)}{q_n(k)}\right).
$$

Call the right-hand side $\mathrm{ELBO}_n(q_n,\boldsymbol\theta)$. Summing over
all $N$ points defines the **Evidence Lower BOund**:

$$
\mathcal{L}(\boldsymbol\theta) \;\ge\; \mathrm{ELBO}(q,\boldsymbol\theta)
\;\equiv\; \sum_{n=1}^{N}\sum_{k=1}^{K} q_n(k)\,\ln\!\left(\frac{\pi_k\,\mathcal{N}(\mathbf{x}_n\mid\boldsymbol\mu_k,\boldsymbol\Sigma_k)}{q_n(k)}\right).
$$

This holds for **every** valid choice of $q$, and for every $\boldsymbol\theta$.
We haven't picked $q$ yet; that's the next move, and it's not arbitrary.

--------------------------------------------------------------------------------

## 2.4  Picking $q$: this *is* the E-step

The bound is only useful if it's *tight*, meaning equal to $\mathcal{L}(\boldsymbol\theta)$
and not just below it. From Section 2.2, Jensen is tight exactly when every
$y_k = \pi_k\mathcal{N}(\mathbf{x}_n\mid\boldsymbol\mu_k,\boldsymbol\Sigma_k)/q_n(k)$
is the *same number* across $k$ (for $k$ with $q_n(k)>0$). Call that common value
$C$. Then $q_n(k) = \pi_k\mathcal{N}(\mathbf{x}_n\mid\boldsymbol\mu_k,\boldsymbol\Sigma_k)/C$
for every $k$, and since the $q_n(k)$ must sum to 1:

$$
1 = \sum_{k=1}^{K} q_n(k) = \frac{1}{C}\sum_{k=1}^{K}\pi_k\,\mathcal{N}(\mathbf{x}_n\mid\boldsymbol\mu_k,\boldsymbol\Sigma_k)
\;\;\Longrightarrow\;\;
C = \sum_{j=1}^{K}\pi_j\,\mathcal{N}(\mathbf{x}_n\mid\boldsymbol\mu_j,\boldsymbol\Sigma_j).
$$

Substituting $C$ back in:

$$
q_n(k) = \frac{\pi_k\,\mathcal{N}(\mathbf{x}_n\mid\boldsymbol\mu_k,\boldsymbol\Sigma_k)}{\sum_{j=1}^{K}\pi_j\,\mathcal{N}(\mathbf{x}_n\mid\boldsymbol\mu_j,\boldsymbol\Sigma_j)} \;=\; r_{nk}.
$$

**The unique choice of $q$ that makes the lower bound touch the true
log-likelihood is exactly the responsibility, computed from the current
parameters.** That is not a heuristic we're borrowing from intuition; it falls
straight out of the equality condition of Jensen's inequality. Computing
$r_{nk}$ from $\boldsymbol\theta^{(t)}$ *is* the **E-step**.

--------------------------------------------------------------------------------

## 2.5  Fixing $q=r$ and maximizing over $\boldsymbol\theta$: this is the M-step

With $q_n(k) = r_{nk}$ now fixed (computed once, from $\boldsymbol\theta^{(t)}$),
expand the ELBO:

$$
\mathrm{ELBO}(\boldsymbol\theta) = \sum_{n=1}^{N}\sum_{k=1}^{K} r_{nk}
\Big[\ln\pi_k + \ln\mathcal{N}(\mathbf{x}_n\mid\boldsymbol\mu_k,\boldsymbol\Sigma_k) - \ln r_{nk}\Big].
$$

The term $-r_{nk}\ln r_{nk}$ does not involve $\boldsymbol\theta$ at all (the
$r_{nk}$ are already fixed numbers), so maximizing over $\boldsymbol\theta$ only
needs:

$$
\underbrace{\sum_{n=1}^{N}\sum_{k=1}^{K} r_{nk}\Big[\ln\pi_k + \ln\mathcal{N}(\mathbf{x}_n\mid\boldsymbol\mu_k,\boldsymbol\Sigma_k)\Big]}_{\text{maximize this over } \pi_k,\ \boldsymbol\mu_k,\ \boldsymbol\Sigma_k}.
$$

Compare this to Section 1.6's complete-data log-likelihood
$\sum_n\sum_k \mathbb{1}[z_n=k]\big(\ln\pi_k+\ln\mathcal{N}(\cdots)\big)$, **identical
in form**, except the hard indicator $\mathbb{1}[z_n=k]$ has been replaced by the
soft responsibility $r_{nk}$. It is a plain **sum**, not a log of a sum, so there's no more
coupling between components. It has closed-form maximizers, which we derive in
Lesson 3 using Lagrange multipliers (for the constrained $\pi_k$) and ordinary
calculus (for $\boldsymbol\mu_k$, $\boldsymbol\Sigma_k$). Finding those maximizers
is the **M-step**.

--------------------------------------------------------------------------------

## 2.6  The theorem: EM never decreases the log-likelihood

Let $\boldsymbol\theta^{(t)}$ be the current parameters.

1. **E-step.** Compute $r_{nk}$ from $\boldsymbol\theta^{(t)}$. By Section 2.4, this
   choice of $q$ makes the bound tight:
   $$\mathrm{ELBO}(\boldsymbol\theta^{(t)}) = \mathcal{L}(\boldsymbol\theta^{(t)}).$$
2. **M-step.** With $q=r$ held fixed, choose $\boldsymbol\theta^{(t+1)}$ to
   *maximize* the ELBO. Since $\boldsymbol\theta^{(t)}$ is one candidate value the
   maximizer is compared against:
   $$\mathrm{ELBO}(\boldsymbol\theta^{(t+1)}) \ge \mathrm{ELBO}(\boldsymbol\theta^{(t)}).$$
3. **Jensen, again, but now generically.** For *any* $\boldsymbol\theta$ and
   *any* $q$ (in particular for $\boldsymbol\theta^{(t+1)}$ with $q$ still $=r$
   from step 1), the bound derived in Section 2.3 holds:
   $$\mathcal{L}(\boldsymbol\theta^{(t+1)}) \ge \mathrm{ELBO}(\boldsymbol\theta^{(t+1)}).$$

Chain the three together:

$$
\mathcal{L}(\boldsymbol\theta^{(t+1)}) \;\ge\; \mathrm{ELBO}(\boldsymbol\theta^{(t+1)}) \;\ge\; \mathrm{ELBO}(\boldsymbol\theta^{(t)}) \;=\; \mathcal{L}(\boldsymbol\theta^{(t)}).
$$

$$
\boxed{\mathcal{L}(\boldsymbol\theta^{(t+1)}) \;\ge\; \mathcal{L}(\boldsymbol\theta^{(t)})}
$$

**Every EM iteration is guaranteed to not decrease the log-likelihood.** It says
nothing about reaching the *global* maximum; that's exactly the gap your RQ1 and
RQ2 experiments will probe.

Be precise about what is and isn't proved. A sequence that never decreases need
not converge, and even when it does, under regularity conditions (Wu, 1983) its
limit is a *stationary point* of $\mathcal{L}$: a local maximum or a saddle. For a
full-covariance GMM, $\mathcal{L}$ is unbounded above (a component collapsing onto
one point), so "always increasing" can run away; the covariance floor built in
Stage 3 guards against this.

In code, this theorem is precisely what you check for
a bug: if you ever see $\mathcal{L}$ *drop* between iterations, the E-step or
M-step has an arithmetic error, not a modeling issue.

```python
# sanity check to run once gmmlab.gmm exists:
lls = []  # log-likelihood after each EM iteration
for t in range(n_iter):
    r = e_step(X, pi, mu, Sigma)
    pi, mu, Sigma = m_step(X, r)
    lls.append(log_likelihood(X, pi, mu, Sigma))
assert all(b >= a - 1e-8 for a, b in zip(lls, lls[1:])), "log-likelihood decreased!"
```

--------------------------------------------------------------------------------

## 2.7  Addendum: the size of the gap, and a link to your uncertainty score

*Read this after you have answered Q1–Q6. It uses only the Jensen route above
plus one new object.*

Define the **KL divergence** between two distributions over the labels,

$$
\mathrm{KL}(q\,\Vert\,p) = \sum_{k=1}^{K} q(k)\,\ln\frac{q(k)}{p(k)} \;\ge\; 0,
\qquad = 0 \iff q = p.
$$

Write $p(\mathbf{x}_n) = p(\mathbf{x}_n, z_n=k)\,/\,p(z_n=k\mid\mathbf{x}_n)$
(product rule), take $\ln$, and average over $k$ with weights $q_n(k)$:

$$
\ln p(\mathbf{x}_n)
= \underbrace{\sum_{k} q_n(k)\ln\frac{p(\mathbf{x}_n, z_n=k)}{q_n(k)}}_{\mathrm{ELBO}_n}
\;+\; \underbrace{\sum_{k} q_n(k)\ln\frac{q_n(k)}{p(z_n=k\mid\mathbf{x}_n)}}_{\mathrm{KL}\left(q_n \Vert\, p(z_n\mid\mathbf{x}_n)\right)}.
$$

Summing over $n$:

$$
\mathcal{L}(\boldsymbol\theta) - \mathrm{ELBO}(q,\boldsymbol\theta)
= \sum_{n=1}^{N} \mathrm{KL}\big(q_n \,\Vert\, p(z_n\mid\mathbf{x}_n,\boldsymbol\theta)\big) \;\ge\; 0.
$$

This is the same bound as before, but now the **gap has a name**. It is zero
exactly when $q_n$ equals the true posterior, which is the responsibility
$r_{nk}$. That is a second proof that the E-step is the right choice: it drives
the gap to zero.

**The link to your project.** Split the ELBO differently:

$$
\mathrm{ELBO}(q,\boldsymbol\theta)
= \underbrace{\sum_{n}\sum_{k} q_n(k)\,\ln\!\big(\pi_k\mathcal{N}(\mathbf{x}_n\mid\boldsymbol\mu_k,\boldsymbol\Sigma_k)\big)}_{\text{expected complete-data log-likelihood}}
\;+\; \sum_{n=1}^{N}\underbrace{\Big(-\sum_{k} q_n(k)\ln q_n(k)\Big)}_{\mathbb{H}[q_n]}.
$$

At the E-step, $q_n = \mathbf{r}_n$, so the second term is exactly
$\sum_n \mathbb{H}[z_n\mid\mathbf{x}_n]$: **the same assignment entropy we will
later use as the uncertainty score for active learning.** EM's own objective
contains it, and the M-step ignores it only because it doesn't depend on
$\boldsymbol\theta$ once $\mathbf{r}$ is fixed.

--------------------------------------------------------------------------------

## Your turn

**Q1: Concavity.**
Show $f(y) = \ln y$ satisfies $f''(y) < 0$ for $y>0$, and state in one sentence
what that implies about the chord between any two points on its graph.

**Q2: Jensen by hand.**
Let $f=\ln$, $y=(2,8)$, $w=(0.25,0.75)$. Compute both sides of Jensen's
inequality by hand (show the arithmetic) and confirm which side is larger.

**Q3: Why multiply and divide by $q_n(k)$?**
If we tried to apply Jensen directly to
$\ln\sum_k \pi_k\mathcal{N}(\mathbf{x}_n\mid\boldsymbol\mu_k,\boldsymbol\Sigma_k)$
*without* introducing $q_n(k)$, what would $w_k$ and $y_k$ have to be? Why does
that choice not give us anything new? (Hint: the most natural split uses the
mixing weights $\pi_k$ as the weights $w_k$. What distribution over $k$ is that,
does it depend on $\mathbf{x}_n$, and when would that bound be tight?)

**Q4: The equality condition, in your own words.**
Explain why forcing
$\pi_k\mathcal{N}(\mathbf{x}_n\mid\boldsymbol\mu_k,\boldsymbol\Sigma_k)/q_n(k)$ to
be the *same number* for every $k$ is exactly what pins $q_n(k)$ down to the
responsibility $r_{nk}$, rather than leaving it free.

**Q5: The monotonicity chain.**
In the three-line chain in Section 2.6, label each of the two $\ge$ steps with which
fact justifies it: "Jensen's inequality" or "the M-step maximizes the ELBO."
Which step would fail if the M-step only found a *worse* $\boldsymbol\theta^{(t+1)}$
than $\boldsymbol\theta^{(t)}$ by mistake?

**Q6: Calibration check.**
Now that you've used $\mathbb{E}_Y[f(Y)] = \sum_k w_k f(y_k)$ concretely in Q2,
re-rate yourself on expectation: `solid` / `shaky` / `new to me`. If still shaky,
say specifically what's unclear, since that tells me what to fold into Lesson 3.
