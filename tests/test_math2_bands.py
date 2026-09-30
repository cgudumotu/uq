"""MATH-2: central intervals use z = inverse-normal((1 + p) / 2); never plus or minus 2 sigma."""

from __future__ import annotations

import numpy as np
import pytest

import uqcalibrate as uqc
from conftest import Z_50, Z_80, Z_90, Z_95
from uqcalibrate import _normal


@pytest.mark.rule("MATH-2")
@pytest.mark.parametrize(("level", "z"), [(0.95, Z_95), (0.90, Z_90), (0.80, Z_80), (0.50, Z_50)])
def test_math2_z_values(level, z):
    assert _normal.z_value(level) == pytest.approx(z, rel=1e-12, abs=0)


@pytest.mark.rule("MATH-2")
def test_math2_never_two_sigma():
    # Points placed at mean + 1.97 std are OUTSIDE a true 95% band (z = 1.96) but would be
    # inside a plus-or-minus-2-sigma band. The package must report 0 coverage here.
    mean = np.zeros(10)
    std = np.ones(10)
    assert uqc.coverage(mean + 1.97 * std, mean, std, level=0.95) == 0.0
    assert uqc.coverage(mean + 1.95 * std, mean, std, level=0.95) == 1.0


@pytest.mark.rule("MATH-2")
@pytest.mark.parametrize("bad", [0.0, 1.0, 95, -0.5, 1.5])
def test_math2_level_out_of_range_raises(bad):
    y = np.zeros(3)
    with pytest.raises(ValueError, match=r"level must be strictly between 0 and 1"):
        uqc.coverage(y, y, np.ones(3), level=bad)
    with pytest.raises(ValueError, match=r"level must be strictly between 0 and 1"):
        uqc.interval_score(y, y, np.ones(3), level=bad)
