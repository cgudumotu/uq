"""The calibration report: every metric together, with a verdict (MISSION MATH-4, MATH-7).

`evaluate` orchestrates: it validates once, calls the metric kernels for every level, applies
the verdict rule, and returns an immutable `CalibrationReport`. It computes no metric itself.
The wording of `str(report)` is fixed by docs/design/UX-COPY.md and is ASCII only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from . import _normal, _validate, metrics

OVERCONFIDENT = "OVERCONFIDENT"
UNDERCONFIDENT = "UNDERCONFIDENT"
CONSISTENT = "CONSISTENT"

HEADLINE_LEVEL = 0.95
_WILSON_Z = _normal.z_value(0.95)  # the interval's own confidence, fixed (HANDOFF D-3)

# Column layout, 79 characters: the verdict column is 14 wide ("UNDERCONFIDENT") and is
# followed by a single space so every row stays under 80 columns (docs/design/UX-COPY.md).
_HEADER = (
    f"{'promised':>8}  {'delivered':>9}  {'plausible range':>15}  {'verdict':<14} "
    f"{'mean width':>10}  {'interval score':>14}"
)
_FOOTER = (
    "plausible range: 95% Wilson interval for the true coverage.\n"
    "A wider range means weaker evidence."
)


@dataclass(frozen=True)
class CalibrationReport:
    """What `evaluate` returns. Immutable; one entry per level in the order of `levels`.

    Attributes
    ----------
    n : int
        Number of points.
    levels : tuple of float
        The promised coverages, as given to `evaluate`.
    delivered : tuple of float
        Delivered coverage at each level, as a fraction in [0, 1].
    plausible_ranges : tuple of (float, float)
        The 95% Wilson interval for the true coverage at each level, as fractions.
    verdicts : tuple of str
        "OVERCONFIDENT", "UNDERCONFIDENT" or "CONSISTENT" at each level.
    mean_widths : tuple of float
        The average band width at each level.
    interval_scores : tuple of float
        The interval score at each level; lower is better.
    nll : float
        The Gaussian negative log-likelihood; lower is better.
    headline : str
        The verdict at 95%, as one sentence.
    """

    n: int
    levels: tuple[float, ...]
    delivered: tuple[float, ...]
    plausible_ranges: tuple[tuple[float, float], ...]
    verdicts: tuple[str, ...]
    mean_widths: tuple[float, ...]
    interval_scores: tuple[float, ...]
    nll: float
    headline: str

    def to_dict(self) -> dict[str, object]:
        """The same numbers as plain Python types (lists, floats, ints, strings)."""
        return {
            "n": self.n,
            "levels": list(self.levels),
            "delivered": list(self.delivered),
            "plausible_ranges": [list(pair) for pair in self.plausible_ranges],
            "verdicts": list(self.verdicts),
            "mean_widths": list(self.mean_widths),
            "interval_scores": list(self.interval_scores),
            "nll": self.nll,
            "headline": self.headline,
        }

    def __str__(self) -> str:
        lines = [f"uqcalibrate: {self.n} points, Gaussian bands", _HEADER]
        rows = zip(
            self.levels,
            self.delivered,
            self.plausible_ranges,
            self.verdicts,
            self.mean_widths,
            self.interval_scores,
            strict=True,
        )
        for level, delivered, (low, high), verdict, width, score in rows:
            promised = f"{level * 100:g}%"
            plausible = f"{low * 100:.1f}% - {high * 100:.1f}%"
            lines.append(
                f"{promised:>8}  {delivered * 100:>8.1f}%  {plausible:>15}  {verdict:<14} "
                f"{width:>10.2f}  {score:>14.2f}"
            )
        lines.append(f"NLL {self.nll:.2f}")
        lines.append(_FOOTER)
        lines.append(self.headline)
        return "\n".join(lines)


def _verdict(low: float, high: float, level: float) -> str:
    """The MATH-7 rule: compare the plausible range for the true coverage with the promise."""
    if high < level:
        return OVERCONFIDENT
    if low > level:
        return UNDERCONFIDENT
    return CONSISTENT


def _headline(verdict: str, level: float, delivered: float) -> str:
    sentence = f"{verdict}: {level * 100:g}% promised, {delivered * 100:.1f}% delivered."
    if verdict == CONSISTENT:
        sentence += " No evidence of miscalibration."
    return sentence


def evaluate(
    y: object,
    mean: object,
    std: object,
    levels: Sequence[float] = (0.5, 0.8, 0.9, 0.95),
) -> CalibrationReport:
    """Check a model's bands against the truth at several levels and return a verdict.

    At each level the report holds the delivered coverage, the plausible range for the true
    coverage (a 95% Wilson interval), a verdict, the mean band width and the interval score;
    plus one Gaussian negative log-likelihood (MATH-4). The verdict at a level is OVERCONFIDENT
    when even the top of the plausible range is below the promise, UNDERCONFIDENT when even the
    bottom is above it, and otherwise CONSISTENT, which means no evidence of a problem at this
    sample size, not proof of calibration (MATH-7). The headline is the verdict at 95%.

    Parameters
    ----------
    y, mean, std
        As in `coverage`.
    levels : sequence of float, default (0.5, 0.8, 0.9, 0.95)
        The promised coverages to check. Must include 0.95.

    Returns
    -------
    CalibrationReport

    Examples
    --------
    >>> report = evaluate(y_test, mean_test, std_test)   # doctest: +SKIP
    >>> print(report)                                    # doctest: +SKIP
    >>> report.headline                                  # doctest: +SKIP
    'OVERCONFIDENT: 95% promised, 65.0% delivered.'
    """
    y_arr, mean_arr, std_arr = _validate.prediction_inputs(y, mean, std)
    level_tuple = _validate.levels(levels)
    if HEADLINE_LEVEL not in level_tuple:
        raise NotImplementedError(
            "evaluate needs 0.95 among levels: the headline verdict is defined at 95% "
            "(MISSION MATH-7), and the headline for other levels is under review. "
            "Add 0.95 to levels."
        )

    n = int(y_arr.size)
    delivered: list[float] = []
    ranges: list[tuple[float, float]] = []
    verdicts: list[str] = []
    widths: list[float] = []
    scores: list[float] = []
    for level in level_tuple:
        z = _normal.z_value(level)
        lower, upper = metrics._bounds(mean_arr, std_arr, z)
        k = int(np.count_nonzero(metrics._inside(y_arr, lower, upper)))
        low, high = _normal.wilson_interval(k, n, _WILSON_Z)
        delivered.append(k / n)
        ranges.append((low, high))
        verdicts.append(_verdict(low, high, level))
        widths.append(metrics._mean_width(std_arr, z))
        scores.append(metrics._interval_score(y_arr, lower, upper, 1.0 - level))

    at_95 = level_tuple.index(HEADLINE_LEVEL)
    return CalibrationReport(
        n=n,
        levels=level_tuple,
        delivered=tuple(delivered),
        plausible_ranges=tuple(ranges),
        verdicts=tuple(verdicts),
        mean_widths=tuple(widths),
        interval_scores=tuple(scores),
        nll=metrics._nll(y_arr, mean_arr, std_arr),
        headline=_headline(verdicts[at_95], HEADLINE_LEVEL, delivered[at_95]),
    )
