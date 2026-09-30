"""MATH-4: every summary reports coverage, width and a proper score together; never
coverage alone.
"""

from __future__ import annotations

import pytest

import uqcalibrate as uqc
from conftest import DEFAULT_LEVELS


@pytest.mark.rule("MATH-4")
def test_math4_evaluate_reports_three_things_per_level(honest):
    y, mean, std = honest
    report = uqc.evaluate(y, mean, std)
    assert report.levels == DEFAULT_LEVELS
    for field in ("delivered", "mean_widths", "interval_scores", "verdicts", "plausible_ranges"):
        assert len(getattr(report, field)) == len(DEFAULT_LEVELS), field
    assert isinstance(report.nll, float)
    assert report.n == len(y)


@pytest.mark.rule("MATH-4")
def test_math4_inflated_sigma_wins_coverage_loses_width_and_nll(honest):
    # A model with sigma x 1000 covers everything and must be seen to pay for it.
    y, mean, std = honest
    honest_report = uqc.evaluate(y, mean, std)
    bloated = uqc.evaluate(y, mean, 1000.0 * std)
    assert bloated.delivered[-1] >= 0.999
    assert bloated.mean_widths[-1] > 100.0 * honest_report.mean_widths[-1]
    assert bloated.nll > honest_report.nll
    assert bloated.interval_scores[-1] > honest_report.interval_scores[-1]
    assert bloated.verdicts[-1] == "UNDERCONFIDENT"
