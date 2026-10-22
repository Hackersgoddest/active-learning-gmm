# Python notes (learned through the project, a little at a time)

The project is the research; Python is a tool we sharpen along the way. So the
rule is small: **every function we build gets a "Python note" of at most four
ideas, each with a 30-second exercise.** Ideas come back as we reuse them, which
is what makes them stick. The order of ideas is dictated by the code we need,
not by a syllabus. Every snippet below was run.

Read `numpy_for_this_project.md` first for the JS/C++ translation table.

--------------------------------------------------------------------------------

## Reading, not writing: by choice, corrected from an earlier plan

An earlier version of this file planned a ladder up to writing functions
yourself. I was explicit about this early on: the priority is understanding what the
code does and why, not personally authoring it: "I don't code from scratch...
but I understand what is going on so in case of any error, I can explain for
the agentic AI to debug." That's a legitimate, different way to work, and it's
exactly what `START_HERE.md`'s one rule already asks for: *"AI may write code.
You must be able to answer these six questions about anything in the repo."*
Writing was never the requirement; explaining was.

So: I write the code. You get the same six-question treatment every function
has had (`gaussian_logpdf`, `logsumexp`, `responsibilities`, the M-step,
`em.fit`, `match_components`): what it computes, why we need it, which
equation it is, what it assumes, how we know it's right, what breaks it.
That's the checkpoint that matters, not who typed the `def`.

--------------------------------------------------------------------------------

## Tools that teach style for you

```
.venv/bin/ruff check .      # style and bug-pattern warnings, with explanations
.venv/bin/ruff format .     # rewrites layout to one consistent style
```

Run both before you consider a piece of code done. The linter is configured in
`pyproject.toml` with a short list of rule families (PEP 8 layout, unused names,
import order, modern syntax, likely bugs, simplifications, NumPy pitfalls). When
it flags something you don't understand, ask, and that becomes a note here.

**A style decision we made, and why.** PEP 8 says variables are `lowercase`. Our
code uses `X`, `Sigma`, `D`, `R`, because they match the notation in
`00_notation.md`, and that is what lets you read the code against the math. The
rule is switched off for those two checks (`N803`, `N806`). Style rules exist to
help the reader; in scientific Python, matching the paper's symbols is accepted
practice (scikit-learn calls the data matrix `X`). Elsewhere we follow PEP 8:
`snake_case` functions, 4-space indent, lines up to 88 characters.

House habits:

- Comments say **why**, not what. The code says what.
- A docstring under every function: one summary line, then `Parameters` and
  `Returns` with shapes.
- Small functions that do one thing, each with a test.
- Error type matches the failure: `ValueError` for bad input, `LinAlgError` when
  the mathematics fails (a matrix that isn't positive definite).

--------------------------------------------------------------------------------

## Note: `gaussian_logpdf`

**1. Guard clauses.** The function checks its inputs first and raises
immediately, then does the real work with no nesting:

```python
if mu.ndim != 1:
    raise ValueError(f"mu must have shape (D,), got {mu.shape}")
```

The alternative, wrapping the whole body in `if valid: ...`, buries the actual
computation. *Exercise:* find the five guards in the file; which raise
`ValueError` and which `LinAlgError`, and why?

**2. f-strings.** `f"... {mu.shape}"` puts a value into the text. A message that
shows the offending value is worth ten that don't.

**3. `_` and tuple unpacking.** `_, logdet = np.linalg.slogdet(Sigma)`:
`slogdet` returns two things; `_` says "I don't need this one". `q, r =
divmod(17, 5)` gives `3, 2`.

**4. Truthiness of arrays (a real trap).** `np.allclose(...)` returns a plain
`bool`, so `if not np.allclose(...)` is fine. But `if some_array:` for an array
with more than one element raises
`ValueError: The truth value of an array with more than one element is
ambiguous. Use a.any() or a.all()`. Python asks for a single yes/no and an array
holds many. *Exercise:* what does `bool(np.array([0]))` give, and why does
`np.array([0, 1])` behave differently?

--------------------------------------------------------------------------------

## Note: `logsumexp`

**1. Default and keyword arguments.** `def logsumexp(a, axis=-1, keepdims=False)`.
Callers may skip the defaults or name them: `logsumexp(A)`, `logsumexp(A,
axis=0)`, `logsumexp(A, keepdims=True)` gave shapes `(2,)`, `(2,)`, `(2, 1)` for
a `(2, 2)` input. Naming arguments at the call site makes calls readable.

**2. A related trap: mutable defaults.**

```python
def add(x, bucket=[]):
    bucket.append(x)
    return bucket

add(1), add(2)      # ([1, 2], [1, 2])  -- the SAME list is reused every call
```

The default is created once, when the function is defined. The linter flags this
(rule `B006`). Use `bucket=None` and create the list inside.

**3. Conditional expression.** `return out if keepdims else np.squeeze(out,
axis=axis)` is the C++/JS `keepdims ? out : squeeze(out)`.

**4. `with`.** `with np.errstate(divide="ignore"):` changes a setting for the
indented block only and **restores it on exit**, even if an error happens, like a
C++ RAII guard. Inside the block `np.log(0.0)` gives `-inf` quietly; outside, the
same call warns `divide by zero encountered in log`. *Exercise:* why did we
silence only `divide` and not everything?

--------------------------------------------------------------------------------

## Note: `responsibilities`

**1. Relative imports, inside the package.**

```python
from .gaussian import gaussian_logpdf      # inside src/gmmlab/responsibilities.py
from .logsumexp import logsumexp
```

The leading `.` means "a sibling module in this same package" (`gmmlab/`). This
is only used *inside* `gmmlab`'s own files. From outside (`tests/`, a future
script) you import the installed package by its full name instead:
`from gmmlab.gaussian import gaussian_logpdf`. The editable install
(`pip install -e .`, done back in Stage 0) is what makes `gmmlab` importable
from anywhere without a relative `.`.

**2. List comprehension building a list of *arrays*, then `np.stack`.**

```python
log_weighted = np.stack(
    [np.log(pi[k]) + gaussian_logpdf(X, mu[k], Sigma[k]) for k in range(k_count)],
    axis=1,
)
```

Read inside-out: the `[... for k in range(k_count)]` makes a Python *list* of
`k_count` arrays, each shape `(N,)`, one per component. `np.stack(..., axis=1)`
then fuses that list into one `(N, K)` array (see the NumPy guide, Section 2, for why
`axis=1` and not `axis=0`). This "build a list, then stack" pattern is how we
turn a per-component loop into one array whenever NumPy has no built-in that
already does the K components at once.

**3. `len(pi)` as the source of truth for K.** No separate `K` argument is
passed in; `k_count = len(pi)` reads it off the one array that's guaranteed to
have exactly K entries. One fewer argument that could disagree with the data.

**4. Docstring cross-references.** The module docstring names the exact lesson
section each formula comes from (`lessons/01_..., Section 1.4`). When you're unsure why
a function looks the way it does, the docstring tells you where to go reread;
treat that as the first thing to check, before asking.

--------------------------------------------------------------------------------

## Note: the M-step (`pi_step`, `mu_step`, `sigma_step`, `m_step`)

**1. Returning, and unpacking, multiple values.** `return pi, mu, sigma` builds
a 3-tuple; the call site `pi, mu, sigma = m_step(X, R)` unpacks it positionally,
same idea as `q, r = divmod(17, 5)` from the `gaussian_logpdf` note. Python
has no named/struct return by default; when a function returns more than two
or three things, or the order gets easy to confuse, that's the usual sign to
switch to a small class instead (see the EM-loop backlog below).

**2. Three small functions, one composing call.** `m_step` is four lines: call
`pi_step`, `mu_step`, `sigma_step`, return the tuple. This is the same shape as
`responsibilities.py` composing `gaussian_logpdf` + `logsumexp`: small,
individually-testable pieces, one thin function gluing them in order. Recognize
the pattern; it is not a new mechanical idea, but it is the main design habit
worth copying into your own code.

**3. Design decisions are part of "style" too, not just syntax.** `Nk_safe` and
`reg_covar` aren't workarounds bolted on to make an error go away; they are
documented, tested engineering choices (see the test file:
`test_dead_component_returns_zero_not_nan`,
`test_reg_covar_keeps_near_singular_component_positive_definite`). Naming the
second one `reg_covar`, matching scikit-learn's `GaussianMixture(reg_covar=...)`
exactly, is also a style choice: a reader who knows that library recognizes it
instantly, and it is what Stage 4's comparison against `sklearn` will line up
against.

--------------------------------------------------------------------------------

## Note: the EM loop (`em.py`)

**1. `@dataclass`.** `GMMFit` bundles `pi`, `mu`, `Sigma`, the log-likelihood
history, and whether it converged into one object, instead of `fit()` returning
a 5-tuple (which note 1 on the M-step flagged as the point to switch). The
`@dataclass` decorator writes `__init__` for you from the type-annotated fields,
so you get `GMMFit(pi=..., mu=..., ...)` and `result.mu` for free, no
boilerplate. Compare to a C++ `struct` with a constructor, or a TS `interface`
plus a class implementing it; Python does both in four lines.

**2. `field(default_factory=list)`, and why not `= []`.** This is the mutable-
default trap from the `logsumexp` note, now showing up as a *dataclass* field
instead of a function argument, same bug, same fix idiom. `field(...)` tells
the dataclass "call `list()` fresh for each new `GMMFit`", rather than sharing
one list across every instance.

**3. `@property`.** `n_iter` is computed from `log_likelihood_history`, not
stored; `result.n_iter` reads like a plain attribute but runs a one-line
function. Use this when a value is always derivable from other fields: storing
it separately risks the two going out of sync.

**4. The `rng if rng is not None else np.random.default_rng()` pattern.** Why
not `rng=None` with `rng = rng or np.random.default_rng()` (the `or` form, used
elsewhere in these notes)? Because `or` falls through on *any* falsy value, and
an `np.random.Generator` instance is always truthy, so that particular case is
actually safe here. The explicit `is not None` is still the more careful habit
in general: it fails only on the one thing you meant to catch (`rng` not
given), not on anything that happens to be falsy.

--------------------------------------------------------------------------------

## Running experiment scripts

Plain script, start to finish: `.venv/bin/python experiments/stage4_sanity_check.py`.
It prints its numeric comparisons to the terminal and saves its figure to
`figures/`; open the PNG from VSCode's file sidebar to look at it. We tried
splitting it into VSCode's inline-runnable `# %%` cells for a faster look
without leaving the editor, but the output was a saved file either way, so it
wasn't worth the extra setup, so we reverted. One file, one way to run it.

We're not using full `.ipynb` notebooks either, and that one's worth keeping
in mind: a notebook's cells can run out of order, leaving hidden state that
doesn't match what's on screen, which fights a project built on reproducible,
seeded results; `.ipynb` is JSON with embedded image output, so git diffs on
it are unreadable, unlike every other file here; and `pytest` imports plain
`.py` modules, not notebooks, the actual reason `gmmlab` could get 41 tests
written against it.

--------------------------------------------------------------------------------

## Note: `matching.py` and `fit_best_of` (added while extending Stage 4)

**1. `max(iterable, key=...)`.** `fit_best_of` picks the best restart with
`max(candidates, key=lambda result: result.log_likelihood_history[-1])`: "the
element of `candidates` for which this function returns the largest value",
not the largest value itself. The C++/JS equivalent is reducing over a
comparator; Python's `max`/`min` take a `key` function instead so you never
write the comparison loop by hand.

**2. Spawning independent sub-seeds from one seed.**
`rng.integers(0, 2**32 - 1, size=n_restarts)` draws `n_restarts` numbers from
the *already-seeded* `rng`, then each becomes `np.random.default_rng(seed)`
for one restart. This is what makes `fit_best_of(X, k, rng=np.random.default_rng(99))`
fully reproducible: same input seed, same 10 restarts, every time, without
the caller having to hand-pick 10 separate seeds themselves.

**3. `scipy.optimize.linear_sum_assignment`, not a hand-written loop.**
`match_components` builds a (K, K) cost matrix (every component of fit A
against every component of fit B) and hands it to `linear_sum_assignment`,
which solves the assignment problem exactly (the Hungarian algorithm) in
polynomial time. The tempting alternative (loop through A's components,
greedily grab each one's nearest unused neighbor in B) is a different
algorithm, and sometimes gives a different, worse answer; see
`tests/test_matching.py::test_beats_greedy_matching` for a constructed case
where it matters. Worth remembering generally: "assignment problem" is a
named, solved problem with a library function; reach for `scipy.optimize`
before writing a matching loop by hand.

--------------------------------------------------------------------------------

## Backlog (introduced when the code first needs them)

`datasets.py` (for RQ1): `numpy.random.Generator` as a function argument
passed through several layers, functions returning several things. Experiment
scripts beyond Stage 4: `pathlib`, saving results, `argparse`. Type hints,
once the function signatures across `gmmlab` settle.
