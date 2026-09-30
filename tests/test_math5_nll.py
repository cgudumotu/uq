"""MATH-5: full Gaussian negative log-likelihood, natural log, mean over points, no clamping."""

from __future__ import annotations

import numpy as np
import pytest

import uqcalibrate as uqc
from conftest import HALF_LOG_2PI, honest_predictions


@pytest.mark.rule("MATH-5")
def test_math5_constant():
    # y equal to the mean with std 1: only the constant half*log(2*pi) remains (P-02: keep it).
    y = np.zeros(4)
    assert uqc.gaussian_nll(y, y, np.ones(4)) == pytest.approx(HALF_LOG_2PI, rel=1e-12)


@pytest.mark.rule("MATH-5")
def test_math5_known_answer_with_residual():
    # residual 2, std 2: half*log(2*pi) + log(2) + 0.5 * (2/2)**2
    y = np.array([2.0])
    expected = HALF_LOG_2PI + np.log(2.0) + 0.5
    assert uqc.gaussian_nll(y, np.zeros(1), np.full(1, 2.0)) == pytest.approx(expected, rel=1e-12)


@pytest.mark.rule("MATH-5")
def test_math5_two_forms_agree(rng):
    # ADR-0005: the standardized-residual form equals the literal MATH-5 formula.
    y, mean, std = honest_predictions(rng, 5_000)
    literal = np.mean(0.5 * np.log(2 * np.pi * std**2) + (y - mean) ** 2 / (2 * std**2))
    assert uqc.gaussian_nll(y, mean, std) == pytest.approx(literal, rel=1e-12)


@pytest.mark.rule("MATH-5")
def test_math5_minimum_at_true_sigma():
    # Scaling the honest std up or down makes the NLL worse; the sweep's minimum is at 1.0.
    rng = np.random.default_rng(5)
    y, mean, std = honest_predictions(rng, 20_000)
    scales = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]
    scores = [uqc.gaussian_nll(y, mean, s * std) for s in scales]
    assert scales[int(np.argmin(scores))] == 1.0


@pytest.mark.rule("MATH-5")
def test_math5_no_clamping_tiny_std_is_scored_not_clipped():
    # A confident and wrong model must be punished, not rescued by a floor on sigma.
    y = np.array([1.0])
    mean = np.zeros(1)
    assert uqc.gaussian_nll(y, mean, np.full(1, 1e-3)) > uqc.gaussian_nll(y, mean, np.full(1, 1e-2))
