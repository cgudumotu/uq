"""MATH-8: one scaling factor fitted on held-out data; the closed-form NLL minimizer (P-04).

s = sqrt(mean(((y - mean) / std) ** 2)): the root-mean-square of the standardized residuals.
"""

from __future__ import annotations

import numpy as np
import pytest

import uqcalibrate as uqc
from conftest import DEFAULT_LEVELS, honest_predictions


@pytest.mark.rule("MATH-8")
def test_math8_known_answer_factor_two():
    # Every residual is exactly twice the reported std, so the bands are half as wide as they
    # should be and the factor is exactly 2.
    mean = np.zeros(4)
    std = np.array([1.0, 2.0, 0.5, 4.0])
    y = mean + 2.0 * std * np.array([1.0, -1.0, 1.0, -1.0])
    assert uqc.fit_scaling(y, mean, std) == pytest.approx(2.0, rel=1e-12)


@pytest.mark.rule("MATH-8")
def test_math8_known_answer_three_four_five():
    # Standardized residuals 3 and 4 give a root-mean-square of sqrt((9 + 16) / 2) = sqrt(12.5).
    y = np.array([3.0, -4.0])
    assert uqc.fit_scaling(y, np.zeros(2), np.ones(2)) == pytest.approx(np.sqrt(12.5), rel=1e-12)


@pytest.mark.rule("MATH-8")
def test_math8_returns_positive_python_float(honest):
    y, mean, std = honest
    s = uqc.fit_scaling(y, mean, std)
    assert type(s) is float
    assert s > 0.0


@pytest.mark.rule("MATH-8")
def test_math8_scaled_residuals_have_unit_rms(honest):
    # The "confession" property: after scaling, the standardized residuals have RMS exactly 1.
    y, mean, std = honest
    s = uqc.fit_scaling(y, mean, std)
    z = (y - mean) / (s * std)
    assert np.sqrt(np.mean(z * z)) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.rule("MATH-8")
def test_math8_is_the_nll_minimizer(honest):
    # On the data it was fitted on, no other factor gives a lower NLL.
    y, mean, std = honest
    s = uqc.fit_scaling(y, mean, std)
    best = uqc.gaussian_nll(y, mean, s * std)
    for factor in (0.5, 0.9, 0.99, 1.01, 1.1, 2.0):
        assert uqc.gaussian_nll(y, mean, factor * s * std) > best


@pytest.mark.rule("MATH-8")
def test_math8_zero_residuals_raise_value_error():
    # Every y equals its mean: no positive finite factor exists (edge case E-d).
    mean = np.linspace(0, 1, 5)
    with pytest.raises(ValueError, match=r"residuals are all zero"):
        uqc.fit_scaling(mean.copy(), mean, np.ones(5))


@pytest.mark.rule("MATH-8")
def test_math8_validates_before_anything_else():
    with pytest.raises(ValueError, match=r"std has 1 value <= 0"):
        uqc.fit_scaling(np.ones(2), np.zeros(2), np.array([1.0, 0.0]))


@pytest.mark.rule("MATH-8")
def test_math8_recovers_factor_two_and_fixes_the_report():
    # The MISSION acceptance test: an overconfident model (std reported at half the truth) gets
    # a factor of about 2 on a calibration split, and a separate test split then reads CONSISTENT.
    rng = np.random.default_rng(8)
    y_cal, mean_cal, std_cal = honest_predictions(rng, 20_000, scale=0.5)
    y_test, mean_test, std_test = honest_predictions(rng, 20_000, scale=0.5)
    s = uqc.fit_scaling(y_cal, mean_cal, std_cal)
    assert s == pytest.approx(2.0, rel=0.02)
    report = uqc.evaluate(y_test, mean_test, s * std_test)
    assert report.verdicts == ("CONSISTENT",) * len(DEFAULT_LEVELS)
