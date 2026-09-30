"""Fix a band that is off by a constant factor (MISSION MATH-8; objective decided by P-04).

The factor is the root-mean-square of the standardized residuals, which is the closed-form
minimizer of the Gaussian negative log-likelihood over a single scale: no optimizer, no
randomness. It is fitted on data disjoint from the data it is evaluated on.
"""

from __future__ import annotations

import numpy as np

from . import _validate


def fit_scaling(y: object, mean: object, std: object) -> float:
    """One factor `s` such that `s * std` is calibrated on held-out data (MATH-8).

    `s = sqrt(mean(((y - mean) / std) ** 2))`: the root-mean-square of the standardized
    residuals. For honest bands these residuals have a spread of 1; if their spread is 2, the
    bands are half as wide as they should be and `s` is 2. This is the factor that minimizes
    the Gaussian negative log-likelihood, and after applying it the standardized residuals
    have a root-mean-square of exactly 1 on the calibration data.

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

    Examples
    --------
    Every residual is twice the reported std, so the factor is 2:

    >>> fit_scaling([2.0, -4.0], [0.0, 0.0], [1.0, 2.0])
    2.0
    """
    y_arr, mean_arr, std_arr = _validate.prediction_inputs(y, mean, std)
    residuals = (y_arr - mean_arr) / std_arr
    if not np.any(residuals):
        raise ValueError(
            "residuals are all zero: every y equals its mean, so no positive finite scaling "
            "factor exists. Fit on a calibration split with real prediction errors."
        )
    return float(np.sqrt(np.mean(residuals * residuals)))
