"""Fix a band that is off by a constant factor (MISSION MATH-8).

The objective that chooses the factor is pending Carol's decision (P-04). Everything else in
MATH-8 is in force and implemented: validation, the refusal when no factor can exist, and the
contract that the factor is fitted on data disjoint from the data it is evaluated on.
"""

from __future__ import annotations

import numpy as np

from . import _validate


def fit_scaling(y: object, mean: object, std: object) -> float:
    """One factor `s` such that `s * std` is calibrated on held-out data (MATH-8).

    Fit it on a calibration split the model was not trained on and will not be evaluated on,
    then apply it as `std_new = s * std` everywhere, including on new data. The function cannot
    detect whether you reused the test split, so keep the splits separate.

    Parameters
    ----------
    y, mean, std
        As in `coverage`: the calibration split's true values, predicted means and predicted
        standard deviations.

    Returns
    -------
    float
        The factor, > 0.

    Raises
    ------
    ValueError
        If the inputs fail validation, or if every residual `y - mean` is zero, because then no
        positive finite factor exists (edge case E-d).
    NotImplementedError
        Until the scaling objective is decided (MISSION section 9, P-04).

    Examples
    --------
    >>> s = fit_scaling(y_cal, mean_cal, std_cal)   # doctest: +SKIP
    >>> report = evaluate(y_test, mean_test, s * std_test)   # doctest: +SKIP
    """
    y_arr, mean_arr, std_arr = _validate.prediction_inputs(y, mean, std)
    residuals = (y_arr - mean_arr) / std_arr
    if not np.any(residuals):
        raise ValueError(
            "residuals are all zero: every y equals its mean, so no positive finite scaling "
            "factor exists. Fit on a calibration split with real prediction errors."
        )
    raise NotImplementedError(
        "fit_scaling is not available yet: the scaling objective (MISSION MATH-8, decision P-04) "
        "is under review. See docs/constitution/MISSION.md, section 9."
    )
