"""Numerically stable ln(sum(exp(a))).

    logsumexp(a) = m + ln( sum_k exp(a_k - m) ),   m = max_k a_k

Exact identity (exp(a_k) = exp(m) * exp(a_k - m)). After the shift the largest
term is exp(0) = 1, so the sum is >= 1: it can neither underflow to 0 nor
overflow to inf. Used for ln p(x_n) = logsumexp_k( ln pi_k + ln N_k(x_n) ) and,
subtracted from the same terms, for the responsibilities (lessons/01, 1.4).
"""

import numpy as np


def logsumexp(a, axis=-1, keepdims=False):
    """ln(sum(exp(a))) along `axis`, without under/overflow.

    A slice that is entirely -inf gives -inf (ln of an empty sum of zeros),
    not nan. Without the `np.isfinite` guard, m = -inf would make a - m =
    (-inf) - (-inf) = nan.
    """
    a = np.asarray(a, dtype=float)
    m = np.max(a, axis=axis, keepdims=True)
    m_safe = np.where(np.isfinite(m), m, 0.0)  # avoid inf - inf
    total = np.sum(np.exp(a - m_safe), axis=axis, keepdims=True)
    with np.errstate(divide="ignore"):  # ln(0) = -inf is intended
        out = np.log(total) + m_safe
    return out if keepdims else np.squeeze(out, axis=axis)
