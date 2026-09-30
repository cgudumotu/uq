"""MATH-6: interval score = width + (2/alpha) * misses, averaged over points; lower is better."""

from __future__ import annotations

import numpy as np
import pytest

import uqcalibrate as uqc
from conftest import Z_95, honest_predictions


@pytest.mark.rule("MATH-6")
def test_math6_inside_scores_width():
    # A point inside the band scores exactly the band width, 2 * z * std.
    mean = np.zeros(3)
    std = np.array([1.0, 2.0, 0.5])
    expected = np.mean(2 * Z_95 * std)
    assert uqc.interval_score(mean, mean, std, level=0.95) == pytest.approx(expected, rel=1e-12)


@pytest.mark.rule("MATH-6")
def test_math6_known_miss():
    # One point, std 1, level 0.95: upper bound is z. A miss of exactly 1 above it costs (2/0.05)*1.
    y = np.array([Z_95 + 1.0])
    expected = 2 * Z_95 + 40.0
    assert uqc.interval_score(y, np.zeros(1), np.ones(1), level=0.95) == pytest.approx(
        expected, rel=1e-12
    )
    # ... and the same miss below the band costs the same.
    y_low = np.array([-Z_95 - 1.0])
    assert uqc.interval_score(y_low, np.zeros(1), np.ones(1), level=0.95) == pytest.approx(
        expected, rel=1e-12
    )


@pytest.mark.rule("MATH-6")
def test_math6_minimum_at_honest_sigma():
    rng = np.random.default_rng(6)
    y, mean, std = honest_predictions(rng, 20_000)
    scales = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]
    scores = [uqc.interval_score(y, mean, s * std, level=0.95) for s in scales]
    assert scales[int(np.argmin(scores))] == 1.0
