"""Combine sampled predictions into one standard deviation per point (MISSION MATH-1).

This is the only place in the package that computes total predictive variance (goal G3).
Every model and every caller uses it (MODEL-2).
"""

from __future__ import annotations

import numpy as np

from . import _validate


def total_std(means: object, noise_vars: object) -> np.ndarray:
    """Total predictive standard deviation by the law of total variance.

    Given S predictive samples (Monte Carlo passes, posterior samples or ensemble members) at
    each of n points, with predicted means `means[s, i]` and predicted noise variances
    `noise_vars[s, i]`::

        Var[i] = mean over s of noise_vars[s, i]   +   variance over s of means[s, i]
                 (aleatoric: the noise the samples report)  (epistemic: how much they disagree)
        std[i] = sqrt(Var[i])

    Variances are added and one square root is taken last. Standard deviations are never
    added. The epistemic term divides by S (population variance): the S samples are the whole
    committee whose average is the prediction, not a sample of a larger one (MATH-1a, P-01), and
    a single sample then contributes exactly zero disagreement.

    Parameters
    ----------
    means : array-like, shape (S, n)
        Predicted means, one row per sample or member. For a single model use `means[None, :]`.
    noise_vars : array-like, shape (S, n)
        Predicted noise *variances* (not standard deviations), each >= 0.

    Returns
    -------
    ndarray, shape (n,)
        The total predictive standard deviation at each point, float64.

    Examples
    --------
    Two members 3 apart, both reporting noise variance 16: sqrt(16 + 9) = 5.

    >>> total_std([[-3.0], [3.0]], [[16.0], [16.0]])
    array([5.])
    """
    means_arr, noise_arr = _validate.sample_inputs(means, noise_vars)
    aleatoric = noise_arr.mean(axis=0)  # average noise variance the samples report
    epistemic = means_arr.var(axis=0)  # disagreement between samples; NumPy's default is ddof=0
    return np.sqrt(aleatoric + epistemic)  # add variances, THEN one square root
