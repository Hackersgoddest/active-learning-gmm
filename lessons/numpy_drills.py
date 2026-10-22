"""Predict-the-shape drills. Run:  .venv/bin/python lessons/numpy_drills.py

For each drill, look at the setup and the expression and work out the SHAPE of
the result in your head. Write your prediction into MY_ANSWERS below:
  - a tuple such as (3, 2) for an array shape,
  - () for a single number,
  - "error" if the expression raises an exception.
Then run the file. It tells you which are right; for wrong ones it shows the
true shape so you can see where your reasoning went off. Leave None to skip.
"""

import numpy as np

MY_ANSWERS = {
    1: None,
    2: None,
    3: None,
    4: None,
    5: None,
    6: None,
    7: None,
    8: None,
    9: None,
    10: None,
    11: None,
    12: None,
}

# (id, setup code, expression)
DRILLS = [
    (1, "X = np.zeros((5, 3))", "X.sum(axis=0)"),
    (2, "X = np.zeros((5, 3))", "X.sum(axis=1, keepdims=True)"),
    (3, "X = np.zeros((5, 3)); mu = np.zeros(3)", "X - mu"),
    (4, "X = np.zeros((5, 3)); mu = np.zeros(5)", "X - mu"),
    (5, "X = np.zeros((5, 3))", "X.T @ X"),
    (6, "X = np.zeros((5, 3))", "X @ X"),
    (7, "X = np.zeros((5, 3))", "X * X"),
    (8, "a = np.zeros((7, 4))", "np.max(a, axis=-1, keepdims=True)"),
    (9, "a = np.zeros((7, 4)); m = np.zeros((7, 1))", "np.sum(np.exp(a - m), axis=-1)"),
    (
        10,
        "X = np.zeros((6, 2)); mu = np.zeros((3, 2))",
        "X[:, None, :] - mu[None, :, :]",
    ),
    (
        11,
        "diff = np.zeros((6, 2)); solved = np.zeros((6, 2))",
        "np.einsum('nd,nd->n', diff, solved)",
    ),
    (
        12,
        "R = np.ones((6, 3)); X = np.ones((6, 2))",
        "(R.T @ X) / R.sum(axis=0)[:, None]",
    ),
]


def truth(setup, expr):
    env = {"np": np}
    exec(setup, env)
    try:
        return np.shape(eval(expr, env))
    except Exception:
        return "error"


def main():
    score = tried = 0
    for i, setup, expr in DRILLS:
        guess = MY_ANSWERS.get(i)
        if guess is None:
            print(f"{i:>2}. skipped     {setup}   ->   {expr}")
            continue
        tried += 1
        real = truth(setup, expr)
        ok = guess == real
        score += ok
        print(f"{i:>2}. {'right' if ok else 'WRONG':<8}  {setup}   ->   {expr}")
        if not ok:
            print(f"      you said {guess!r}; the real answer is {real!r}")
    print(f"\n{score}/{tried} attempted correct ({len(DRILLS) - tried} skipped)")


if __name__ == "__main__":
    main()
