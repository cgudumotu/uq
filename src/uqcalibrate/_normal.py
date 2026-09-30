"""Normal-distribution arithmetic from the standard library (MISSION MATH-2, MATH-7; ADR-0004).

Pure math, no policy: the verdict rule that uses `wilson_interval` lives in `report.py`.
Both functions assume validated arguments; the public functions validate first.
"""

from __future__ import annotations

import math
from statistics import NormalDist

_STANDARD_NORMAL = NormalDist()


def z_value(level: float) -> float:
    """The half-width, in standard deviations, of the central band at `level`.

    z = inverse-normal((1 + level) / 2). For level 0.95 this is 1.959964, not 2.

    Examples
    --------
    >>> round(z_value(0.95), 6)
    1.959964
    """
    return _STANDARD_NORMAL.inv_cdf((1.0 + level) / 2.0)


def wilson_interval(k: int, n: int, z: float) -> tuple[float, float]:
    """The Wilson score interval for a proportion, from k successes out of n.

    Defined at k = 0 and k = n, unlike the normal approximation; that is why MATH-7 uses it.
    `z` is the quantile for the interval's own confidence (1.96 for 95%).

    Examples
    --------
    >>> low, high = wilson_interval(183, 200, 1.959964)
    >>> round(low, 3), round(high, 3)
    (0.868, 0.946)
    """
    p_hat = k / n
    z_sq_over_n = z * z / n
    denominator = 1.0 + z_sq_over_n
    center = (p_hat + z_sq_over_n / 2.0) / denominator
    half_width = z * math.sqrt(p_hat * (1.0 - p_hat) / n + z_sq_over_n / (4.0 * n)) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)
