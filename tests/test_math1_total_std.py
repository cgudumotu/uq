"""MATH-1: total predictive variance = mean noise variance + variance of the means (divisor S)."""

from __future__ import annotations

import numpy as np
import pytest

import uqcalibrate as uqc

# The 3-4-5 case from MISSION MATH-1: two members 3 apart around 0, both reporting noise
# variance 16.
MEANS_345 = [[-3.0], [3.0]]
NOISE_345 = [[16.0], [16.0]]


@pytest.mark.rule("MATH-1")
def test_math1_three_four_five():
    # epistemic variance of [-3, 3] with divisor S=2 is 9; aleatoric mean is 16; sqrt(25) = 5.
    np.testing.assert_allclose(uqc.total_std(MEANS_345, NOISE_345), [5.0], rtol=1e-12)


@pytest.mark.rule("MATH-1")
def test_math1_three_four_five_is_location_invariant():
    # Shifting both members by 10 changes nothing: only the spread matters.
    np.testing.assert_allclose(uqc.total_std([[7.0], [13.0]], NOISE_345), [5.0], rtol=1e-12)


@pytest.mark.rule("MATH-1")
def test_math1_two_member_unequal_noise():
    # means 1 and 3: population variance 1. noise variances 2 and 4: mean 3. sqrt(1 + 3) = 2.
    np.testing.assert_allclose(uqc.total_std([[1.0], [3.0]], [[2.0], [4.0]]), [2.0], rtol=1e-12)


@pytest.mark.rule("MATH-1")
def test_math1_single_member_returns_sqrt_noise():
    # S = 1: no disagreement is possible, so the result is the noise std alone (MATH-1a, P-01).
    out = uqc.total_std([[0.5, -1.0]], [[4.0, 9.0]])
    np.testing.assert_allclose(out, [2.0, 3.0], rtol=1e-12)


@pytest.mark.rule("MATH-1")
def test_math1_rejects_thesis_formula():
    # The thesis line: sqrt(std(mu, ddof=1)) + sqrt(mean(var)). On the 3-4-5 inputs it gives 6.0598.
    means = np.asarray(MEANS_345)
    noise = np.asarray(NOISE_345)
    thesis = np.sqrt(means.std(axis=0, ddof=1)) + np.sqrt(noise.mean(axis=0))
    np.testing.assert_allclose(thesis, [6.059767143907118], rtol=1e-12)  # the wrong number
    ours = uqc.total_std(MEANS_345, NOISE_345)
    assert not np.allclose(ours, thesis)
    np.testing.assert_allclose(ours, [5.0], rtol=1e-12)


@pytest.mark.rule("MATH-1")
def test_math1_divisor_is_population():
    # Five members 9, 10, 10, 10, 11 with no noise: variance 2/5 = 0.4 (not 2/4 = 0.5).
    means = [[9.0], [10.0], [10.0], [10.0], [11.0]]
    noise = [[0.0]] * 5
    np.testing.assert_allclose(uqc.total_std(means, noise), [np.sqrt(0.4)], rtol=1e-12)
    assert not np.isclose(uqc.total_std(means, noise)[0], np.sqrt(0.5))


@pytest.mark.rule("MATH-1")
def test_math1_matches_formula_on_random_inputs(rng):
    means = rng.normal(size=(7, 40))
    noise = rng.uniform(0.1, 2.0, size=(7, 40))
    expected = np.sqrt(noise.mean(axis=0) + means.var(axis=0))  # ddof=0 is NumPy's default
    out = uqc.total_std(means, noise)
    assert out.shape == (40,)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, expected, rtol=1e-12)


@pytest.mark.rule("MATH-1")
def test_math1_accepts_lists_and_integers():
    out = uqc.total_std([[1, 3]], [[0, 0]])
    np.testing.assert_allclose(out, [0.0, 0.0])
