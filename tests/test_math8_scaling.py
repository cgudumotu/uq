"""MATH-8: one scaling factor fitted on held-out data. The objective is pending (P-04).

The in-force clauses are tested now: validation, the all-zero-residuals refusal, and the fact
that the pending path raises NotImplementedError naming the rule. The acceptance test is written
too, marked xfail(strict=True): when P-04 is decided and the formula lands, it will XPASS, the
suite will fail, and the marker must be removed. That is the intended reminder.
"""

from __future__ import annotations

import numpy as np
import pytest

import uqcalibrate as uqc
from conftest import DEFAULT_LEVELS, honest_predictions


@pytest.mark.rule("MATH-8")
def test_math8_pending_raises_not_implemented(honest):
    y, mean, std = honest
    with pytest.raises(NotImplementedError, match=r"MATH-8.*P-04"):
        uqc.fit_scaling(y, mean, std)


@pytest.mark.rule("MATH-8")
def test_math8_zero_residuals_raise_value_error():
    # Every y equals its mean: no positive finite factor exists (edge case E-d). Checked before
    # the pending path, so this clause is in force today.
    mean = np.linspace(0, 1, 5)
    with pytest.raises(ValueError, match=r"residuals are all zero"):
        uqc.fit_scaling(mean.copy(), mean, np.ones(5))


@pytest.mark.rule("MATH-8")
def test_math8_validates_before_anything_else():
    with pytest.raises(ValueError, match=r"std has 1 value <= 0"):
        uqc.fit_scaling(np.ones(2), np.zeros(2), np.array([1.0, 0.0]))


@pytest.mark.rule("MATH-8")
@pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="P-04 pending Carol's decision")
def test_math8_recovers_factor_two_and_fixes_the_report():
    rng = np.random.default_rng(8)
    y_cal, mean_cal, std_cal = honest_predictions(rng, 20_000, scale=0.5)
    y_test, mean_test, std_test = honest_predictions(rng, 20_000, scale=0.5)
    s = uqc.fit_scaling(y_cal, mean_cal, std_cal)
    assert s == pytest.approx(2.0, rel=0.02)
    report = uqc.evaluate(y_test, mean_test, s * std_test)
    assert report.verdicts == ("CONSISTENT",) * len(DEFAULT_LEVELS)
