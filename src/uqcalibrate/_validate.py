"""Input checks shared by every public function (MISSION API-2; ADR-0003).

One module owns every check and every message, so each message is written once and tested
once (docs/design/UX-COPY.md). Each function returns clean float64 arrays or plain Python
values; the callers never re-check what came back.

Every message has the same three parts, in this order: what is wrong, with the number that
proves it; what is required; the likely cause and the fix as code.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real

import numpy as np


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _as_float_array(value: object, name: str) -> np.ndarray:
    """Convert anything array-like to a float64 array, or explain why that is impossible."""
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} could not be converted to numbers ({exc}). Pass an array of floats."
        ) from exc


def _check_finite(arr: np.ndarray, name: str) -> None:
    finite = np.isfinite(arr)
    if finite.all():
        return
    bad = np.flatnonzero(~finite.ravel())
    raise ValueError(
        f"{name} has {_plural(int(bad.size), 'non-finite value')} (NaN or inf), first at index "
        f"{int(bad[0])}. All inputs must be finite. Remove or fill those points before "
        "evaluating; uqcalibrate never drops them silently."
    )


def _as_1d(value: object, name: str) -> np.ndarray:
    """A float64 array of shape (n,), with a message for the usual column-vector mistake."""
    arr = _as_float_array(value, name)
    if arr.ndim == 2 and arr.shape[1] == 1:
        raise ValueError(
            f"{name} has shape {arr.shape} but must have shape (n,): one value per point. "
            f"Use {name}.ravel() if it has one column."
        )
    if arr.ndim != 1:
        raise ValueError(
            f"{name} has shape {arr.shape} but must have shape (n,): one value per point."
        )
    return arr


def _mean_1d(mean: object, n: int) -> np.ndarray:
    """`mean` as a (n,) array; a (S, n) array gets the "samples passed as a mean" message."""
    arr = _as_float_array(mean, "mean")
    if arr.ndim == 2 and arr.shape[1] == n and arr.shape[0] != 1:
        raise ValueError(
            f"mean has shape {arr.shape} but y has shape ({n},); mean must have shape (n,). "
            "If these are S samples per point, combine them first: mean = means.mean(axis=0); "
            "std = uqcalibrate.total_std(means, noise_vars)."
        )
    return _as_1d(arr, "mean")


def prediction_inputs(
    y: object, mean: object, std: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate (y, mean, std) for every metric and for the report.

    Returns float64 arrays of shape (n,), with `std` broadcast from a scalar if one was given.
    """
    y_arr = _as_1d(y, "y")
    if y_arr.size == 0:
        raise ValueError("y is empty. At least one point is required.")
    n = int(y_arr.size)

    mean_arr = _mean_1d(mean, n)
    if mean_arr.shape != y_arr.shape:
        raise ValueError(
            f"mean has shape {mean_arr.shape} but y has shape {y_arr.shape}; they must match, "
            "one value per point."
        )

    std_raw = _as_float_array(std, "std")
    if std_raw.ndim == 0:
        std_arr = np.full(n, float(std_raw))
    else:
        std_arr = _as_1d(std_raw, "std")
        if std_arr.shape != y_arr.shape:
            raise ValueError(
                f"std has shape {std_arr.shape} but y has shape {y_arr.shape}; std must have "
                "shape (n,) or be one number."
            )

    _check_finite(y_arr, "y")
    _check_finite(mean_arr, "mean")
    _check_finite(std_arr, "std")

    not_positive = std_arr <= 0
    if not_positive.any():
        raise ValueError(
            f"std has {_plural(int(not_positive.sum()), 'value')} <= 0 (min {std_arr.min():g}). "
            "Standard deviations must be positive. If these are variances or log-variances, "
            "convert them first, e.g. std = np.sqrt(var)."
        )
    return y_arr, mean_arr, std_arr


def sample_inputs(means: object, noise_vars: object) -> tuple[np.ndarray, np.ndarray]:
    """Validate the (S, n) pair that `total_std` combines (MISSION section 6 shapes)."""
    means_arr = _as_float_array(means, "means")
    if means_arr.ndim != 2:
        raise ValueError(
            f"means has shape {means_arr.shape} but must have shape (S, n): S samples or members "
            "on axis 0, n points on axis 1. For a single model use means[None, :], which gives "
            "S = 1."
        )
    if means_arr.size == 0:
        raise ValueError("means is empty. At least one sample and one point are required.")

    noise_arr = _as_float_array(noise_vars, "noise_vars")
    if noise_arr.ndim != 2:
        raise ValueError(
            f"noise_vars has shape {noise_arr.shape} but must have shape (S, n), the same as "
            "means. For a single model use noise_vars[None, :], which gives S = 1."
        )
    if noise_arr.shape != means_arr.shape:
        raise ValueError(
            f"noise_vars has shape {noise_arr.shape} but means has shape {means_arr.shape}; "
            "they must match."
        )

    _check_finite(means_arr, "means")
    _check_finite(noise_arr, "noise_vars")

    negative = noise_arr < 0
    if negative.any():
        raise ValueError(
            f"noise_vars has {_plural(int(negative.sum()), 'value')} < 0 "
            f"(min {noise_arr.min():g}). "
            "Noise variances must be >= 0. If these are standard deviations, square them first: "
            "noise_vars = std**2."
        )
    return means_arr, noise_arr


def level(p: object) -> float:
    """One band level, strictly between 0 and 1 (MISSION MATH-2)."""
    if isinstance(p, bool) or not isinstance(p, Real):
        raise ValueError(
            f"level must be a number strictly between 0 and 1 (got {p!r}). "
            "A 95% band is level=0.95."
        )
    if not 0.0 < float(p) < 1.0:
        raise ValueError(
            f"level must be strictly between 0 and 1 (got {p}). A 95% band is level=0.95."
        )
    return float(p)


def levels(values: object) -> tuple[float, ...]:
    """The levels a report covers: a non-empty sequence of distinct valid levels."""
    is_sequence = isinstance(values, Sequence) and not isinstance(values, (str, bytes))
    is_array = isinstance(values, np.ndarray) and values.ndim == 1
    if not (is_sequence or is_array):
        raise ValueError(
            f"levels must be a sequence of numbers, e.g. levels=(0.9, 0.95) (got {values!r})."
        )
    if len(values) == 0:
        raise ValueError("levels is empty. Give at least one level, e.g. levels=(0.9, 0.95).")
    out: list[float] = []
    for value in values:
        p = level(value)
        if p in out:
            raise ValueError(f"levels has a duplicate ({p:g}). Each level appears once.")
        out.append(p)
    return tuple(out)
