# Matplotlib for this project

Same deal as `numpy_for_this_project.md`: only what we actually use, each line
verified by running it. Every RQ from here on produces at least one plot, so
this is worth the 20 minutes.

--------------------------------------------------------------------------------

## 1. The two objects: Figure and Axes

A **Figure** is the whole image (what gets saved to a file). An **Axes** is one
plot inside it: the thing with x/y data on it. You almost always get both at
once:

```python
fig, ax = plt.subplots()                 # one Figure, one Axes
fig, axes = plt.subplots(2, 2)            # one Figure, a (2, 2) GRID of Axes
axes.shape                                # (2, 2) -- it's a NumPy array of Axes objects
axes[0, 0].scatter(...)                   # draw on the top-left one specifically
```

`experiments/stage4_sanity_check.py` uses `plt.subplots(2, 2, figsize=(11, 10))`
for its 4-panel figure, then draws on `axes[0, 0]`, `axes[0, 1]`, etc.
`figsize=(11, 10)` is the saved image size in inches.

--------------------------------------------------------------------------------

## 2. The plotting calls we use

```python
ax.scatter(x[:, 0], x[:, 1], c=labels, cmap="coolwarm", s=10, alpha=0.5)
```

Points, not lines. `c=labels` colours each point by a number (here, which true
cluster it came from); `cmap` picks the colour scale; `s` is marker size;
`alpha` is transparency (useful when points overlap).

```python
ax.plot(ll_history, marker="o")
```

A line through a sequence of y-values (x is just `0, 1, 2, ...` unless you give
it explicitly). `marker="o"` puts a dot at each actual data point, not just the
line. That's worth doing whenever there are few enough points to want them visible
individually, which is almost always true for an EM log-likelihood trace.

```python
ax.set_title("..."); ax.set_xlabel("..."); ax.set_ylabel("...")
ax.legend()                # shows labels passed as label="..." to scatter/plot
ax.grid(alpha=0.3)         # faint gridlines
ax.set_aspect("equal")     # one unit on x = one unit on y -- essential for
                            # a covariance ellipse to actually look like its
                            # real shape, instead of being stretched by
                            # whatever the axes happened to auto-scale to
```

--------------------------------------------------------------------------------

## 3. Drawing a covariance ellipse (the one non-obvious piece)

This is lessons/01, Section 1.2's geometry, turned into pixels:

```python
from matplotlib.patches import Ellipse

eigvals, eigvecs = np.linalg.eigh(cov)                       # ascending order
angle = np.degrees(np.arctan2(eigvecs[1, -1], eigvecs[0, -1]))   # largest eigenvector's angle
width, height = 2 * n_std * np.sqrt(eigvals[::-1])            # diameters, largest axis first
ax.add_patch(Ellipse(xy=mean, width=width, height=height, angle=angle, fill=False))
```

`np.linalg.eigh` returns eigenvalues smallest-to-largest, with the matching
eigenvectors as **columns** of `eigvecs` in the same order, so `eigvecs[:, -1]`
is the eigenvector for the *largest* eigenvalue, i.e. the ellipse's long axis.
`eigvecs[1, -1]` and `eigvecs[0, -1]` are that eigenvector's y- and x-components;
`arctan2(y, x)` gives its angle. `Ellipse` wants a *diameter*, not a radius, and
*degrees*, not radians, hence `2 *` and `np.degrees(...)`.

`n_std=2.0` is a plotting choice (draw 2 standard deviations out along each
axis), not a precise "95% of the probability is inside this ellipse" claim;
that would need a chi-squared quantile, not just `n_std`. Fine for a
sanity-check plot; would matter for a figure making a quantitative claim.

--------------------------------------------------------------------------------

## 4. Saving

```python
fig.tight_layout()                      # shrinks subplots so titles/labels don't overlap
fig.savefig("figures/name.png", dpi=130)  # dpi = resolution; higher = bigger file, sharper image
```

Always call `tight_layout()` before `savefig` on a multi-panel figure: skip it
and labels routinely collide with the next panel.

--------------------------------------------------------------------------------

## 5. Reading a covariance-ellipse plot (not a matplotlib fact, but needed to
use one)

- **Ellipse center** = that component's fitted mean $\boldsymbol\mu_k$.
- **Ellipse orientation and elongation** = that component's fitted
  $\boldsymbol\Sigma_k$: round means roughly spherical covariance (like our
  "easy case", generated with $\boldsymbol\Sigma=\mathbf{I}$); tilted and
  stretched means correlated, unequal-variance covariance (the "harder case";
  compare the three differently-shaped ellipses to the three different
  `true_covs` matrices in the script that generated them).
- Two ellipses from **different sources overlapping almost exactly** (the
  bottom-right panel) is a strong visual correctness check: two
  independently-written pieces of code agreeing isn't a coincidence. Drawn as
  a thick, translucent band (`gmmlab`) with a thin, opaque line (`sklearn`) on
  top, not as two same-width lines in different colours: when two things are
  expected to coincide almost exactly, two equal-width lines just blend
  together right where the plot most needs to be legible. A thin line staying
  inside a thick band is easy to confirm at a glance; the same wouldn't be
  true of two overlapping hairlines.
