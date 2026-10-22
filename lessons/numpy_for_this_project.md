# Python + NumPy for this project

For a reader coming from JavaScript/TypeScript and C++, who directs an AI to
write code and needs to **read it, predict what it does, and explain a failure
precisely**. Not a course: only what the build stage uses. Every output below was
produced by running the code.

How to use it: open a terminal in the project and run
`.venv/bin/python -i` (an interactive prompt). Type each snippet, and **before you
press Enter on a shape-related line, say the shape out loud**. Then do the drills
in `lessons/numpy_drills.py`. About an hour in total.

--------------------------------------------------------------------------------

## 1. Python in ten minutes (JS/C++ translation)

| you know | Python | note |
|---|---|---|
| `{ ... }` blocks | indentation | the colon `:` opens a block |
| `const x = 1;` | `x = 1` | no declaration, no `;`, dynamic types |
| `null` | `None` | |
| `&&  \|\|  !` | `and  or  not` | |
| `Math.pow(a,b)` / `a**b` | `a ** b` | `//` is integer division |
| `// comment` | `# comment` | `"""docstring"""` under a `def` documents it |
| `function f(a, b = 1)` | `def f(a, b=1):` | |
| destructuring `[a, b] = f()` | `a, b = f()` | `_` means "I don't need this": `_, logdet = slogdet(S)` |
| `arr.map(f)` | `[f(k) for k in range(K)]` | list comprehension; `sum(f(k) for k in range(K))` sums a generator |
| `` `${x}` `` | `f"{x}"` | |
| `import x from "y"` | `import numpy as np` / `from y import x` | |
| `throw new Error()` | `raise ValueError("...")` | `try: ... except ValueError as e:` |
| `arr.slice(1, 3)` | `a[1:3]` | negative index counts from the end: `a[-1]` is the last |
| `a === b` (reference) | `a is b` | `a == b` compares values |

Two traps for a C++/JS programmer:

- **Arrays are passed by reference.** `b = a` does not copy. Use `b = a.copy()`.
- **`*` on arrays is element-by-element, not matrix multiplication.** Matrix
  multiplication is `@`. Confusing them gives a wrong answer, not always an error.

There is no compile step and no static types. **`.shape` is your type system**:
the habit below is what replaces the compiler.

--------------------------------------------------------------------------------

## 2. What a NumPy array is

Think of a C++ `std::vector<double>` (or a JS `Float64Array`) that also carries a
**shape**. Operations run as loops in compiled C, which is why "vectorized" code
is fast. Everything is `float64` unless told otherwise.

```python
import numpy as np
X = np.array([[1., 2.], [3., 4.], [5., 6.]])
X.shape                      # (3, 2)   3 rows (N points), 2 columns (D features)
```

**`axis` names the dimension that disappears.**

```python
X.sum(axis=0)                # [ 9. 12.]     shape (2,)   collapsed the rows
X.sum(axis=1)                # [ 3.  7. 11.] shape (3,)   collapsed the columns
X.sum(axis=1, keepdims=True) # shape (3, 1)  the collapsed axis stays, with length 1
```

`keepdims=True` matters because a `(3, 1)` result can be broadcast back against
`X` (next section). `logsumexp` uses exactly this.

Transpose and matrix product:

```python
X.T.shape                    # (2, 3)
(X @ X.T).shape              # (3, 3)
(X.T @ X).shape              # (2, 2)
X * X                        # element-by-element squares, shape (3, 2)  NOT a matrix product
```

**`np.stack`: turning a list of same-shape arrays into one bigger array**, along
a new axis you choose.

```python
rows = [np.array([1., 2., 3.]), np.array([10., 20., 30.])]   # two (3,) arrays
np.stack(rows, axis=0).shape   # (2, 3): stacked as new ROWS
np.stack(rows, axis=1).shape   # (3, 2): stacked as new COLUMNS
np.stack(rows, axis=1)
# [[ 1. 10.]
#  [ 2. 20.]
#  [ 3. 30.]]
```

`responsibilities.py` does
`np.stack([per_component_result for k in range(K)], axis=1)`: it has a list of
K arrays, each shape `(N,)` (one per component), and wants one `(N, K)` array
with component as the *column*. `axis=1` is what makes that happen; `axis=0`
would instead give `(K, N)`.

--------------------------------------------------------------------------------

## 3. Broadcasting (the one idea that makes NumPy compact)

Rule: line the shapes up **from the right**. Two dimensions are compatible if
they are equal or one of them is 1 (the 1 is stretched). A missing leading
dimension counts as 1.

```python
mu = np.array([1., 1.])      # shape (2,)
X - mu                       # (3,2) - (2,)  ->  (3,2): mu is subtracted from every row
```

This is the line `diff = X - mu` in `gaussian_logpdf`.

```python
col = np.array([[10.], [20.], [30.]])   # shape (3,1)
X + col                      # (3,2) + (3,1) -> (3,2): each row gets its own number added
X - np.array([1., 2., 3.])   # ValueError: operands could not be broadcast together
                             # with shapes (3,2) (3,)     <- fails LOUDLY: 2 vs 3
X - mu.reshape(1, 2)         # shape (3,2), works fine <- this is how my old bug stayed silent
```

Adding a length-1 axis with `None` is how the M-step code pairs every point with
every component:

```python
A = np.zeros((3, 2)); B = np.zeros((4, 2))
(A[:, None, :] - B[None, :, :]).shape   # (3, 4, 2): every row of A minus every row of B
```

**`np.maximum`: clamping, element by element.** (Not `np.max`, which reduces an
array to one number; `np.maximum` compares two things element-wise and keeps
the larger.)

```python
n_k = np.array([5.0, 0.0, 3.0])
np.maximum(n_k, 1e-10)       # [5. 1e-10 3.] -- the 0 is floored, the rest untouched
```

This is how `mstep.py` avoids `0/0 = nan` when a component's responsibility is
zero everywhere: floor the *denominator* only, never the quantity itself
(`pi_k = N_k/N` still uses the real, un-floored `N_k`, so it's still exactly 0
when it should be).

--------------------------------------------------------------------------------

## 4. Indexing, masks, `np.where`

```python
X[:, 0]                      # [1. 3. 5.]  first column (":" means "all rows")
X[-1]                        # [5. 6.]     last row
X[1:]                        # rows 1 and onward
X[X > 2]                     # [3. 4. 5. 6.]  a boolean mask picks elements
a = np.array([3., -np.inf, 1.])
np.isfinite(a)               # [ True False  True]
np.where(np.isfinite(a), a, 0.0)   # [3. 0. 1.]  "a where finite, else 0"
```

The last line is the guard in `logsumexp`: if the max is `-inf`, replace it by 0
so we never compute `(-inf) - (-inf) = nan`.

--------------------------------------------------------------------------------

## 5. `einsum`: a loop written as a formula

`np.einsum('nd,nd->n', diff, solved)` reads as a loop: the letters are loop
indices. A letter that appears in the inputs but **not** after `->` is summed
over. So this means

```python
[ sum(diff[n, d] * solved[n, d] for d in range(D)) for n in range(N) ]   # the loop
(diff * solved).sum(axis=1)                                              # same thing
```

All three give the same numbers (`[0.5, 6.5, 20.5]` in my check). In our code it
computes `(x_n - mu)^T Sigma^-1 (x_n - mu)` for every point at once.

--------------------------------------------------------------------------------

## 6. The three `np.linalg` calls we use

```python
S = np.array([[2., 0.], [0., 4.]])
np.linalg.solve(S, np.array([2., 4.]))   # [1. 1.]   solves S x = b (no explicit inverse)
np.linalg.slogdet(S)                     # sign=1.0, logabsdet=2.0794 = ln(8) = ln(det S)
np.linalg.eigvalsh(S)                    # [2. 4.]   eigenvalues of a symmetric matrix, ascending
```

You already know what each means mathematically; this just shows the call shape.

--------------------------------------------------------------------------------

## 7. Random numbers (you'll need reproducible seeds for research)

```python
rng = np.random.default_rng(0)           # seed 0
rng.normal(size=(2, 2))                  # [[ 0.1257 -0.1321] [ 0.6404  0.1049]]
```

The same seed gives the same numbers every run, which is what lets an experiment
be repeated and a result be trusted.

--------------------------------------------------------------------------------

## 8. Reading NumPy errors (so you can hand them to an AI precisely)

Read a traceback **from the bottom**: the last line is the error, the lines above
say where. All of these were reproduced:

| you see | what it means |
|---|---|
| `ValueError: operands could not be broadcast together with shapes (3,2) (3,)` | shapes don't line up from the right; print both `.shape`s |
| `ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0 ... (size 3 is different from 2)` | inner dimensions of `@` differ: `(3,2) @ (3,2)` needs one transposed |
| `numpy.linalg.LinAlgError: Singular matrix` | the matrix has no inverse (e.g. all zeros); for us, a degenerate covariance |
| `AttributeError: 'list' object has no attribute 'shape'` | it's a Python list, not an array: wrap in `np.asarray(...)` |
| `IndexError: index 5 is out of bounds for axis 0 with size 2` | indexing past the end |
| `RuntimeWarning: overflow encountered in exp` | result became `inf` (not an error, easy to miss) |
| `RuntimeWarning: divide by zero encountered in log` | `log(0)` is `-inf` |
| `RuntimeWarning: invalid value encountered in log` | `log` of a negative number gave `nan` |

Warnings are the dangerous ones: the program keeps running with `inf`/`nan`. Also,
`np.array([1.0, np.nan]).max()` returns `nan` with **no warning at all**.

**How to report a bug to an AI:** the failing input's shapes, the full traceback
(or the wrong number and the number you expected), and what you already checked.
"It gives nan" is not enough; "`R` has shape (100, 3) and row 17 is all nan;
`Sigma[1]` has eigenvalues [1e-18, 2.3]" is.

**How to ask an AI for NumPy code:** specify by shape and math, not by syntax.
For example: *"Compute `R` of shape (N, K) where `R[n, k]` is the responsibility
of component k for point n, in log-space, and add a test against a plain loop."*

--------------------------------------------------------------------------------

## 9. Your loop-first habit, adjusted

Loop version = the readable specification; vectorized version = the fast one.
For each remaining function I'll write both, and a test asserts they agree.
Your job is to read the loop version like C++, then use the shape trace to see
that the vectorized one is the same computation. Here is `logsumexp` both ways:

```python
def logsumexp_loop(a):                      # a: list of numbers; C++-style spec
    m = max(a)
    total = 0.0
    for x in a:
        total += math.exp(x - m)
    return m + math.log(total)
```

```python
def logsumexp(a, axis=-1, keepdims=False):  # a: (..., K)
    a = np.asarray(a, dtype=float)
    m = np.max(a, axis=axis, keepdims=True)                 # (..., 1)   largest per slice
    m_safe = np.where(np.isfinite(m), m, 0.0)               # (..., 1)   guard against -inf
    total = np.sum(np.exp(a - m_safe), axis=axis, keepdims=True)   # (..., 1)  broadcast subtract
    out = np.log(total) + m_safe                            # (..., 1)
    return out if keepdims else np.squeeze(out, axis=axis)  # drop the length-1 axis
```

Shape trace for a `(N, K)` input with `axis=-1`: `a` is `(N,K)`; `m` is `(N,1)`;
`a - m_safe` is `(N,K)` by broadcasting; `total` is `(N,1)`; the result is `(N,)`.

Now do `lessons/numpy_drills.py`.
