# Review of your Lesson 2 answers

*Feedback from my AI tutor on my answers to the Lesson 2 questions, kept as a record of what I got wrong and how I fixed it.*

Three of six are correct or nearly so (Q1, Q3,
Q6). Q2, Q4, Q5 have real gaps that matter for Lesson 3, so we stay in LEARN
mode; please redo the three flagged items before we move on.

--------------------------------------------------------------------------------

## Q1: correct, one precision fix

$f''(y) = -1/y^2$, and for every $y>0$ this is **strictly** negative (never
zero), so $\ln$ is *strictly* concave, not just "concave". Your conclusion is
right: the chord sits at or below the curve (concave means the curve bulges
above the chord). No re-answer needed.

--------------------------------------------------------------------------------

## Q2: the formula was applied wrong; please redo this one

Jensen's inequality is
$$
f\Big(\sum_k w_k y_k\Big) \;\ge\; \sum_k w_k\,f(y_k),
$$
i.e. **take the weighted average of $y_k$, then apply $f$** (left side) versus
**apply $f$ to each $y_k$, then take the weighted average** (right side).

You computed the right side as $\ln(w_1 y_1) + \ln(w_2 y_2)$; that is neither
side of the inequality, it's $\ln$ applied to $w_k y_k$ and then summed, not
$w_k \cdot \ln(y_k)$ summed.

Correct arithmetic for $f=\ln$, $y=(2,8)$, $w=(0.25,0.75)$:

$$
\text{LHS} = \ln(0.25\cdot2 + 0.75\cdot8) = \ln(6.5) \approx 1.8718
$$

$$
\text{RHS} = 0.25\ln(2) + 0.75\ln(8) = 0.25(0.6931) + 0.75(2.0794) \approx 0.1733+1.5596 = 1.7329
$$

$1.8718 \ge 1.7329$, confirmed, correctly this time. Your version happened to
land on $\ln 3 \approx 1.0986$, a coincidence of the numbers, not a validation.

**Please redo this by hand and paste the numbers back**, since this arithmetic *is*
the expectation operation ($\mathbb{E}_Y[f(Y)]$), which is exactly what Q6 asks
you to self-rate on.

--------------------------------------------------------------------------------

## Q3: good instinct, one precision fix, no re-answer needed

You correctly identified $w_k=\pi_k$, $y_k=\mathcal{N}(\mathbf{x}_n\mid\boldsymbol\mu_k,\boldsymbol\Sigma_k)$,
and correctly saw the real reason this choice is useless: $\pi_k$ doesn't depend
on $\mathbf{x}_n$, so it can't adapt per point. That's the main point.

One correction: the equality condition is that $y_k$ **alone** must be equal
across $k$, not "weight times density". Since
$y_k = \pi_k\mathcal{N}_k(\mathbf{x}_n)/q_n(k)$ and here $q_n(k)=\pi_k$, the
$\pi_k$'s cancel, leaving $y_k=\mathcal{N}_k(\mathbf{x}_n)$. So the condition is
literally "every component assigns $\mathbf{x}_n$ the *same density*", true
only by coincidence, and it says nothing about where $\mathbf{x}_n$ actually sits
relative to the components. That's the sharp version of your own point.

--------------------------------------------------------------------------------

## Q4: right conclusion, mechanism missing; please restate

Your answer states the outcome but not why it's *forced*. The derivation:

1. Equality requires $y_k = \dfrac{\pi_k\mathcal{N}_k(\mathbf{x}_n)}{q_n(k)}$ to
   equal the **same constant** $C$ for every $k$.
2. Rearranged: $q_n(k) = \pi_k\mathcal{N}_k(\mathbf{x}_n)/C$ for every $k$. So
   $q_n(k)$ is already **pinned to a fixed shape**, proportional to
   $\pi_k\mathcal{N}_k(\mathbf{x}_n)$, with only $C$ left unknown.
3. $q_n$ must be a valid distribution: $\sum_k q_n(k)=1$. That is one more
   equation, and with one unknown $C$ it has exactly one solution:
   $C=\sum_j\pi_j\mathcal{N}_j(\mathbf{x}_n)$.

Proportionality (step 2) plus normalization (step 3) leaves **zero degrees of
freedom**: there is exactly one $q_n$ satisfying both, and substituting $C$
back gives exactly $r_{nk}$. "It sums to 1" isn't a side note, it's the second
half of what pins the answer down uniquely.

**Please restate this back in your own words.** The M-step derivation in
Lesson 3 uses the exact same "constrained optimum, pinned by normalization"
logic for $\pi_k$, so this is worth having solid.

--------------------------------------------------------------------------------

## Q5: the labeling and the failure mode are both off; corrected version

$$
\mathcal{L}(\theta^{(t+1)}) \;\underbrace{\ge}_{\text{(A)}}\; \mathrm{ELBO}(\theta^{(t+1)}) \;\underbrace{\ge}_{\text{(B)}}\; \mathrm{ELBO}(\theta^{(t)}) \;=\; \mathcal{L}(\theta^{(t)})
$$

- **(A)** is **Jensen's inequality**: it holds for *any* $\theta$ and *any* valid
  $q$, unconditionally. It never fails, because it isn't a claim about our
  algorithm; it's a general mathematical fact.
- **(B)** is **"the M-step maximizes the ELBO"**: $\theta^{(t+1)}$ is chosen as
  the maximizer with $q=r$ fixed, and $\theta^{(t)}$ is just one candidate value
  being compared against, so the max can't be worse.
- The equality on the right ($\mathrm{ELBO}(\theta^{(t)})=\mathcal{L}(\theta^{(t)})$)
  is not one of the two $\ge$ steps; it's the E-step tightness from Section 2.4,
  already established before this chain starts.

If the M-step had a bug and returned a worse $\theta^{(t+1)}$: **(B) fails, not
(A)**. Jensen's inequality is true regardless of what your code does; it
cannot "break". A buggy M-step just means $\mathrm{ELBO}(\theta^{(t+1)})$ could
come out below $\mathrm{ELBO}(\theta^{(t)})$, which snaps the chain, and the
guarantee $\mathcal{L}(\theta^{(t+1)})\ge\mathcal{L}(\theta^{(t)})$ is gone.
That's exactly why the lesson said: a likelihood that decreases in your code is
always an E-step or M-step bug, never a failure of the math itself.

**Please relabel this in your own words**: which step is Jensen, which is the
M-step, and what breaks.

--------------------------------------------------------------------------------

## Q6

Given the Q2 slip, hold "solid" loosely until Q2 is redone correctly, since that
arithmetic *is* the expectation operation in disguise.

--------------------------------------------------------------------------------

## What to send back

Three things (plain text is fine, no LaTeX needed: write `ln`, `sqrt`,
`pi_k`, etc.):

1. Q2 redone with the correct formula.
2. Q4's mechanism in your own words (proportionality, then normalization).
3. Q5 relabeled: which step is Jensen, which is the M-step, what breaks if the
   M-step has a bug.
