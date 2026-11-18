"""Small, dependency-free resampling statistics for EXP-L.

Intervals are bootstrap percentile intervals from explicit seeds.  Paired
comparisons resample the paired unit (seed/example), matching the design.
"""

from __future__ import annotations

import math
from typing import Sequence

from .determinism import make_rng


def bootstrap_ci(values: Sequence[float], seed: int, n_boot: int = 2000,
                 alpha: float = 0.05) -> dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    rng = make_rng(seed, "bootstrap-ci")
    n = len(vals)
    means = []
    for _ in range(n_boot):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo_i = int((alpha / 2) * n_boot)
    hi_i = min(n_boot - 1, int((1 - alpha / 2) * n_boot))
    return {"mean": sum(vals) / n, "lo": means[lo_i], "hi": means[hi_i], "n": n}


def paired_bootstrap_ci(a: Sequence[float], b: Sequence[float], seed: int,
                        n_boot: int = 2000, alpha: float = 0.05) -> dict[str, float]:
    """CI for mean(a) - mean(b), paired by index."""
    if len(a) != len(b) or not a:
        return {"mean_diff": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    diffs = [float(x) - float(y) for x, y in zip(a, b)]
    return bootstrap_ci(diffs, seed, n_boot, alpha)


def sign_test(a: Sequence[float], b: Sequence[float]) -> dict[str, float | int]:
    """Exact two-sided binomial sign test on paired differences."""
    pos = sum(1 for x, y in zip(a, b) if x > y)
    neg = sum(1 for x, y in zip(a, b) if x < y)
    n = pos + neg
    if n == 0:
        return {"n_pos": pos, "n_neg": neg, "n": n, "p_two_sided": 1.0}
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {"n_pos": pos, "n_neg": neg, "n": n,
            "p_two_sided": min(1.0, 2 * tail)}
