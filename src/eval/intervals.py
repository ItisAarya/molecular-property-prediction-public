"""
src/eval/intervals.py

The one confidence-interval function, and the one t-table behind it.

    from src.eval.intervals import ci95
    mean, sd, half, n = ci95(values)

WHY THIS EXISTS
---------------
There were four. `src/eval/stats.py`, `src/eval/view_stats.py`, `src/eval/ece_multiseed.py`
and `scripts/make_tables.py` each carried their own `ci95` and their own `T_CRIT` dict, and
the dicts had drifted: two stopped at n = 6 and two ran to n = 8. Above six observations the
short tables silently fell back to 1.96 -- the normal approximation -- while the long ones
used the correct t-value, so the same five numbers could have produced a 2.365 interval in
one table and a 1.96 interval in another.

**No published number was affected.** Every caller aggregates over the five seeded splits,
n = 5, where all four tables agreed on 2.776. The divergence was latent: it would have
appeared the first time anyone added a sixth and seventh seed, in a paper whose argument is
that this project measures more carefully than the field does.

So the definition lives here, once, and the t-table goes far enough that the fallback is not
reachable by any plausible run.

THE RETURN VALUE INCLUDES n ON PURPOSE
--------------------------------------
`n` is the count of *finite* observations, after filtering -- which is not `len(values)` when
a split is missing a result. Callers need it to tell "no data" from "one observation", and
the old four-way split disagreed about that too: one returned `None`, one `nan`, one `0.0`.
Returning the count lets each caller decide, explicitly, rather than inferring it from a
sentinel.

Half-width is `nan` for fewer than two observations, never `0.0`. A single measurement has
no interval, and reporting "+/- 0.0000" for one observation claims a precision that does not
exist -- which is the kind of thing this project exists to object to.
"""

import numpy as np

# Two-sided 95% critical values of Student's t, by number of observations (df = n - 1).
# Runs to 20 so the 1.96 fallback is unreachable for any realistic multi-seed run.
T_CRIT = {
    2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365,
    9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145,
    16: 2.131, 17: 2.120, 18: 2.110, 19: 2.101, 20: 2.093,
}


def t_critical(n):
    """The two-sided 95% t multiplier for `n` observations; 1.96 only beyond the table."""
    return T_CRIT.get(n, 1.96)


def ci95(values):
    """
    Mean, standard deviation, 95% t-interval half-width, and the number of finite values.

    `None` and non-finite entries are dropped before anything is computed, so a split that
    produced no result lowers `n` rather than poisoning the mean with a NaN.

    Returns `(nan, nan, nan, 0)` when nothing is left, and `(mean, 0.0, nan, 1)` for a
    single observation -- see the module docstring on why the half-width is not 0.0.
    """
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    n = int(v.size)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    if n == 1:
        return float(v[0]), 0.0, np.nan, 1
    sd = float(v.std(ddof=1))
    return float(v.mean()), sd, t_critical(n) * sd / np.sqrt(n), n
