"""MATH-3: coverage is the fraction of points inside the band, boundaries inclusive."""

from __future__ import annotations

import numpy as np
import pytest

import uqcalibrate as uqc
from conftest import DEFAULT_LEVELS, Z_95, honest_predictions


@pytest.mark.rule("MATH-3")
def test_math3_all_inside_all_outside_boundary():
    mean = np.linspace(-1, 1, 5)
    std = np.full(5, 0.5)
    assert uqc.coverage(mean, mean, std) == 1.0  # every point at its own mean
    assert uqc.coverage(mean + 100, mean, std) == 0.0  # every point far outside
    # A point exactly on the upper bound counts as inside (inclusive boundaries).
    on_bound = mean + Z_95 * std
    assert uqc.coverage(on_bound, mean, std, level=0.95) == 1.0
    on_lower = mean - Z_95 * std
    assert uqc.coverage(on_lower, mean, std, level=0.95) == 1.0


@pytest.mark.rule("MATH-3")
def test_math3_returns_python_float_fraction(honest):
    y, mean, std = honest
    c = uqc.coverage(y, mean, std)
    assert type(c) is float
    assert 0.0 <= c <= 1.0


@pytest.mark.rule("MATH-3")
def test_math3_oracle_reaches_nominal_coverage():
    # Honest bands on 100,000 points: delivered coverage within 0.005 of the promise.
    # One SD of a coverage estimate at p = 0.5 and n = 100,000 is 0.0016, so 0.005 is ~3 SD.
    rng = np.random.default_rng(0)
    y, mean, std = honest_predictions(rng, 100_000)
    for level in DEFAULT_LEVELS:
        assert abs(uqc.coverage(y, mean, std, level=level) - level) < 0.005


@pytest.mark.rule("MATH-3")
def test_math3_scalar_std_broadcasts():
    y = np.array([0.0, 0.1, -0.1, 5.0])
    mean = np.zeros(4)
    assert uqc.coverage(y, mean, 1.0) == 0.75
