"""Scores for a Gaussian predictive band (MISSION MATH-2, MATH-3, MATH-5, MATH-6).

Each metric exists once, as a private kernel that trusts its inputs. The public functions
validate through `_validate` and then call one kernel; `report.evaluate` validates once and
calls the kernels directly (ADR-0003). Kernels are never exported.
"""

from __future__ import annotations

import math

import numpy as np

from . import _normal, _validate

HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)  # 0.9189385332046727, the MATH-5 constant (P-02)


# --- kernels (trusted inputs: float64 arrays of shape (n,), std > 0, 0 < level < 1) -------------


def _bounds(mean: np.ndarray, std: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray]:
    """The central band [mean - z*std, mean + z*std] (MATH-2)."""
    half = z * std
    return mean - half, mean + half


def _inside(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Boolean mask of points inside the band, boundaries inclusive (MATH-3)."""
    return (lower <= y) & (y <= upper)


def _nll_points(y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Full Gaussian negative log-likelihood of every point (MATH-5, ADR-0005).

    Computed as half*log(2*pi) + log(std) + half*z**2 with z = (y - mean) / std, which equals
    half*log(2*pi*std**2) + (y - mean)**2 / (2*std**2) without ever squaring std. `compare`
    uses the per-point values for its paired test (MODEL-5a); `_nll` averages them.
    """
    z = (y - mean) / std
    return HALF_LOG_2PI + np.log(std) + 0.5 * z * z


def _nll(y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    """Full Gaussian negative log-likelihood, averaged over points (MATH-5)."""
    return float(np.mean(_nll_points(y, mean, std)))


def _interval_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> float:
    """Width plus (2/alpha) times the distance of each miss, averaged over points (MATH-6)."""
    width = upper - lower
    below = np.maximum(lower - y, 0.0)  # positive only where y < lower
    above = np.maximum(y - upper, 0.0)  # positive only where y > upper
    return float(np.mean(width + (2.0 / alpha) * (below + above)))


def _mean_width(std: np.ndarray, z: float) -> float:
    """The average band width, 2*z*std (the sharpness term of MATH-4)."""
    return float(np.mean(2.0 * z * std))


# --- public functions ----------------------------------------------------------------------------


def coverage(y: object, mean: object, std: object, level: float = 0.95) -> float:
    """The fraction of true values inside the central band at `level` (MATH-3).

    The band is `mean +/- z*std` with `z = inverse-normal((1 + level) / 2)` (MATH-2); for
    level 0.95, z = 1.959964. Boundaries count as inside.

    Parameters
    ----------
    y : array-like, shape (n,)
        The true values.
    mean : array-like, shape (n,)
        The predicted means.
    std : array-like, shape (n,), or one number
        The predicted standard deviations, each > 0.
    level : float, default 0.95
        The band's promised coverage, strictly between 0 and 1.

    Returns
    -------
    float
        Delivered coverage as a fraction in [0, 1]; reports print it as a percentage.

    Examples
    --------
    >>> coverage([0.0, 0.1, -0.1, 5.0], [0.0, 0.0, 0.0, 0.0], 1.0)
    0.75
    """
    y_arr, mean_arr, std_arr = _validate.prediction_inputs(y, mean, std)
    p = _validate.level(level)
    lower, upper = _bounds(mean_arr, std_arr, _normal.z_value(p))
    return float(_inside(y_arr, lower, upper).mean())


def gaussian_nll(y: object, mean: object, std: object) -> float:
    """The Gaussian negative log-likelihood, averaged over points (MATH-5). Lower is better.

    A "surprise score": how surprised the model is by the truth, given the mean and standard
    deviation it claimed. It includes the constant half*log(2*pi) so values compare with the
    published literature (MATH-5a), and it never clamps `std`.

    Parameters
    ----------
    y, mean, std
        As in `coverage`.

    Returns
    -------
    float

    Examples
    --------
    With y equal to the mean and std 1, only the constant remains:

    >>> round(gaussian_nll([1.0, 2.0], [1.0, 2.0], 1.0), 6)
    0.918939
    """
    y_arr, mean_arr, std_arr = _validate.prediction_inputs(y, mean, std)
    return _nll(y_arr, mean_arr, std_arr)


def interval_score(y: object, mean: object, std: object, level: float = 0.95) -> float:
    """The interval score of the central band at `level` (MATH-6). Lower is better.

    For each point: the band's width, plus (2 / (1 - level)) times the distance by which the
    true value misses the band, if it does (Gneiting and Raftery, 2007, section 6.2). Averaged
    over points. Widening the band raises the width term; narrowing it raises the miss term, so
    neither trick improves the score.

    Parameters
    ----------
    y, mean, std, level
        As in `coverage`.

    Returns
    -------
    float

    Examples
    --------
    A point inside the band scores exactly the width, 2 * 1.959964 * 1:

    >>> round(interval_score([0.0], [0.0], 1.0), 6)
    3.919928
    """
    y_arr, mean_arr, std_arr = _validate.prediction_inputs(y, mean, std)
    p = _validate.level(level)
    lower, upper = _bounds(mean_arr, std_arr, _normal.z_value(p))
    return _interval_score(y_arr, lower, upper, 1.0 - p)
