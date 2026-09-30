"""MATH-7: Wilson-interval verdict per level; headline at 95%; the evidence shown in every row."""

from __future__ import annotations

import numpy as np
import pytest

import uqcalibrate as uqc
from conftest import DEFAULT_LEVELS, Z_95, honest_predictions
from uqcalibrate import _normal


def points_with_k_inside(k: int, n: int):
    """n points with mean 0 and std 1; exactly k inside every band (y = 0), the rest far out."""
    y = np.where(np.arange(n) < k, 0.0, 100.0)
    return y, np.zeros(n), np.ones(n)


@pytest.mark.rule("MATH-7")
@pytest.mark.parametrize(
    ("k", "n", "low", "high"),
    [
        (183, 200, 0.868104, 0.946254),
        (186, 200, 0.885945, 0.957848),
        (0, 10, 0.0, 0.277533),
        (200, 200, 0.981155, 1.0),
    ],
)
def test_math7_wilson_known_answers(k, n, low, high):
    got_low, got_high = _normal.wilson_interval(k, n, Z_95)
    assert got_low == pytest.approx(low, abs=1e-6)
    assert got_high == pytest.approx(high, abs=1e-6)


@pytest.mark.rule("MATH-7")
def test_math7_verdict_thresholds_at_n_200():
    # 183/200 = 91.5%: even the top of the plausible range (94.6%) is below 95% -> OVERCONFIDENT.
    y, mean, std = points_with_k_inside(183, 200)
    report = uqc.evaluate(y, mean, std)
    assert report.verdicts[-1] == "OVERCONFIDENT"
    assert report.headline == "OVERCONFIDENT: 95% promised, 91.5% delivered."
    # 186/200 = 93.0%: the range reaches 95.8%, so there is no evidence of a problem -> CONSISTENT.
    y, mean, std = points_with_k_inside(186, 200)
    report = uqc.evaluate(y, mean, std)
    assert report.verdicts[-1] == "CONSISTENT"
    assert report.headline == (
        "CONSISTENT: 95% promised, 93.0% delivered. No evidence of miscalibration."
    )
    assert report.plausible_ranges[-1] == pytest.approx((0.885945, 0.957848), abs=1e-6)


@pytest.mark.rule("MATH-7")
def test_math7_underconfident_headline():
    # Everything inside at every level: at 95% the range [98.1%, 100%] lies above the promise.
    y, mean, std = points_with_k_inside(200, 200)
    report = uqc.evaluate(y, mean, std)
    assert report.verdicts == ("UNDERCONFIDENT",) * 4
    assert report.headline == "UNDERCONFIDENT: 95% promised, 100.0% delivered."


@pytest.mark.rule("MATH-7")
def test_math7_oracle_consistent_at_large_n():
    rng = np.random.default_rng(0)
    y, mean, std = honest_predictions(rng, 100_000)
    report = uqc.evaluate(y, mean, std)
    assert report.verdicts == ("CONSISTENT",) * len(DEFAULT_LEVELS)


@pytest.mark.rule("MATH-7")
def test_math7_halved_sigma_overconfident(rng):
    y, mean, std = honest_predictions(rng, 2_000, scale=0.5)
    report = uqc.evaluate(y, mean, std)
    assert report.verdicts == ("OVERCONFIDENT",) * len(DEFAULT_LEVELS)


@pytest.mark.rule("MATH-7")
def test_math7_small_n_consistent_unless_extreme():
    # n = 10 with 9 or 10 inside at 95%: the plausible range is wide and contains 95%.
    for k in (9, 10):
        y, mean, std = points_with_k_inside(k, 10)
        assert uqc.evaluate(y, mean, std).verdicts[-1] == "CONSISTENT"
    # Known property of the rule, not a bug: 8 of 10 inside gives a range topping out at
    # 94.3%, which is below 95%, so an honest model is flagged here about 8.6% of the time.
    y, mean, std = points_with_k_inside(8, 10)
    assert uqc.evaluate(y, mean, std).verdicts[-1] == "OVERCONFIDENT"
    # An extreme miss at n = 10 is still called: all 10 inside a 50% band is UNDERCONFIDENT.
    y, mean, std = points_with_k_inside(10, 10)
    assert uqc.evaluate(y, mean, std).verdicts[0] == "UNDERCONFIDENT"


@pytest.mark.rule("MATH-7")
def test_math7_headline_needs_95_raises_not_implemented(honest):
    y, mean, std = honest
    with pytest.raises(NotImplementedError, match=r"MATH-7"):
        uqc.evaluate(y, mean, std, levels=(0.5, 0.9))


@pytest.mark.rule("MATH-7")
def test_math7_report_text_is_ascii_and_narrow(honest):
    y, mean, std = honest
    text = str(uqc.evaluate(y, mean, std))
    assert text.isascii()
    lines = text.splitlines()
    assert all(len(line) < 80 for line in lines)
    assert lines[0] == "uqcalibrate: 2000 points, Gaussian bands"
    assert lines[1].startswith("promised  delivered  plausible range  verdict")
    assert "plausible range: 95% Wilson interval for the true coverage." in text
    assert "A wider range means weaker evidence." in text
    assert lines[-1].startswith(("OVERCONFIDENT", "UNDERCONFIDENT", "CONSISTENT"))


@pytest.mark.rule("MATH-7")
def test_math7_report_row_format():
    y, mean, std = points_with_k_inside(183, 200)
    lines = str(uqc.evaluate(y, mean, std)).splitlines()
    # Golden row: width = 2 * 1.959964 = 3.92; score = width + 17 misses of 98.04 at 40x, over 200.
    assert lines[1] == (
        "promised  delivered  plausible range  verdict        mean width  interval score"
    )
    assert lines[5] == (
        "     95%      91.5%    86.8% - 94.6%  OVERCONFIDENT        3.92          337.26"
    )
