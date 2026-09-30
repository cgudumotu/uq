"""Compare several models' bands on the same test data (MISSION MODEL-5a, MODEL-3, MATH-4).

`compare` ranks the models that report total uncertainty by their Gaussian negative
log-likelihood, shows coverage and width beside it so no model is ever judged by coverage alone,
and names a winner only when its lead over the runner-up is larger than two standard errors of
the paired per-point difference. Models whose bands hold no noise term are listed in their own
section and are never ranked (MODEL-3). The M2 function `train_and_compare` calls this; the
ranking rule exists once, here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from . import _validate, metrics
from .report import CalibrationReport, evaluate

_EPISTEMIC_HEADER = "Epistemic only (bands hold no noise term; not ranked)"


@dataclass(frozen=True)
class ModelComparison:
    """What `compare` returns. Immutable.

    Attributes
    ----------
    n : int
        Number of test points.
    names : tuple of str
        The ranked models, best first (lowest NLL first).
    reports : tuple of CalibrationReport
        One report per ranked model, aligned with `names`.
    best : str or None
        The leader's name when it beats the runner-up by more than two standard errors of the
        paired per-point NLL difference (or when it is the only ranked model); otherwise None.
    runner_up : str or None
        The second-ranked model, when there are at least two.
    lead : float or None
        Mean per-point NLL advantage of the leader over the runner-up.
    lead_se : float or None
        Standard error of that mean difference (paired, over the n points).
    epistemic_only_names : tuple of str
        Models without a noise term, in the order given; shown, never ranked.
    epistemic_only_reports : tuple of CalibrationReport
        Their reports, aligned with `epistemic_only_names`.
    headline : str
        One sentence naming the winner, or saying the top two are too close to call.
    """

    n: int
    names: tuple[str, ...]
    reports: tuple[CalibrationReport, ...]
    best: str | None
    runner_up: str | None
    lead: float | None
    lead_se: float | None
    epistemic_only_names: tuple[str, ...]
    epistemic_only_reports: tuple[CalibrationReport, ...]
    headline: str

    def to_dict(self) -> dict[str, object]:
        """The same content as plain Python types; each report as its own `to_dict()`."""
        return {
            "n": self.n,
            "names": list(self.names),
            "reports": [report.to_dict() for report in self.reports],
            "best": self.best,
            "runner_up": self.runner_up,
            "lead": self.lead,
            "lead_se": self.lead_se,
            "epistemic_only_names": list(self.epistemic_only_names),
            "epistemic_only_reports": [report.to_dict() for report in self.epistemic_only_reports],
            "headline": self.headline,
        }

    def __str__(self) -> str:
        all_names = self.names + self.epistemic_only_names
        width = max([len("model"), *(len(name) for name in all_names)])
        header = (
            f"{'rank':>4}  {'model':<{width}}  {'95% delivered':>13}  {'mean width':>10}  "
            f"{'NLL':>7}  verdict"
        )

        def row(label: str, name: str, report: CalibrationReport) -> str:
            at_95 = report.levels.index(0.95)
            return (
                f"{label:>4}  {name:<{width}}  {report.delivered[at_95] * 100:>12.1f}%  "
                f"{report.mean_widths[at_95]:>10.2f}  {report.nll:>7.2f}  {report.verdicts[at_95]}"
            )

        lines = [
            f"uqcalibrate: {len(self.names)} models ranked, "
            f"{len(self.epistemic_only_names)} epistemic-only, {self.n} points",
            header,
        ]
        for rank, (name, report) in enumerate(zip(self.names, self.reports, strict=True), 1):
            lines.append(row(str(rank), name, report))
        lines.append(self.headline)
        if self.epistemic_only_names:
            lines.append("")
            lines.append(_EPISTEMIC_HEADER)
            for name, report in zip(
                self.epistemic_only_names, self.epistemic_only_reports, strict=True
            ):
                lines.append(row("-", name, report))
        return "\n".join(lines)


def _checked_mapping(value: object, argument: str) -> dict[str, tuple[object, object]]:
    """A mapping of name -> (mean, std), with the API-2 messages for the usual mistakes."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{argument} must be a mapping from model name to (mean, std), e.g. "
            f"{{'my model': (mean, std)}} (got {type(value).__name__})."
        )
    out: dict[str, tuple[object, object]] = {}
    for name, pair in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"model names must be non-empty strings (got {name!r}).")
        if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)) or len(pair) != 2:
            count = len(pair) if isinstance(pair, Sequence) and not isinstance(pair, str) else 1
            raise ValueError(
                f"{argument}[{name!r}] must be a pair (mean, std) (got {count} items)."
            )
        out[name] = (pair[0], pair[1])
    return out


def _headline(
    names: tuple[str, ...], best: str | None, lead: float | None, lead_se: float | None
) -> str:
    if not names:
        return "No model reports total uncertainty; nothing ranked."
    if len(names) == 1:
        return f"Best: {names[0]} (the only model ranked)."
    if best is not None:
        return f"Best: {best} (leads the runner-up by {lead:.2f} +/- {lead_se:.2f} NLL per point)."
    return f"Too close to call: the top two differ by {lead:.2f} +/- {lead_se:.2f} NLL per point."


def compare(
    y: object,
    predictions: Mapping[str, tuple[object, object]],
    epistemic_only: Mapping[str, tuple[object, object]] | None = None,
    levels: Sequence[float] = (0.5, 0.8, 0.9, 0.95),
) -> ModelComparison:
    """Rank several models' bands on the same test data (MODEL-5a).

    Every model gets the full `evaluate` report. The models in `predictions` (those reporting
    total uncertainty) are ordered by NLL, lowest first; coverage and mean width at 95% are
    shown beside it (MATH-4). The leader is named best only when its per-point NLL beats the
    runner-up's by more than two standard errors of the paired difference; otherwise the top two
    are reported as too close to call. Models in `epistemic_only` (bands without a noise term)
    are reported in their own section and never ranked (MODEL-3).

    Pass predictions *after* any scaling factor from `fit_scaling` has been applied, and use
    test data that no factor was fitted on.

    Parameters
    ----------
    y : array-like, shape (n,)
        The true values; n >= 2.
    predictions : mapping of str to (mean, std)
        Models that report total uncertainty. `mean` has shape (n,); `std` has shape (n,) or is
        one number. May be empty only if `epistemic_only` is not.
    epistemic_only : mapping of str to (mean, std), optional
        Models whose bands hold no noise term.
    levels : sequence of float, default (0.5, 0.8, 0.9, 0.95)
        As in `evaluate`; must include 0.95.

    Returns
    -------
    ModelComparison

    Examples
    --------
    >>> y = [0.0, 0.0, 0.0]
    >>> result = compare(y, {"wide": (y, 2.0), "tight": (y, 1.0)})
    >>> result.names
    ('tight', 'wide')
    >>> result.headline
    'Best: tight (leads the runner-up by 0.69 +/- 0.00 NLL per point).'
    """
    ranked_inputs = _checked_mapping(predictions, "predictions")
    epistemic_inputs = _checked_mapping(epistemic_only, "epistemic_only")
    if not ranked_inputs and not epistemic_inputs:
        raise ValueError(
            "compare needs at least one model: both predictions and epistemic_only are empty."
        )
    shared = sorted(set(ranked_inputs) & set(epistemic_inputs))
    if shared:
        raise ValueError(
            f"{shared[0]!r} appears in both predictions and epistemic_only; a model is one or "
            "the other."
        )

    def score(pair: tuple[object, object]) -> tuple[CalibrationReport, np.ndarray]:
        y_arr, mean_arr, std_arr = _validate.prediction_inputs(y, pair[0], pair[1])
        if y_arr.size < 2:
            raise ValueError(
                "compare needs at least 2 points to estimate the lead's standard error "
                f"(got {y_arr.size})."
            )
        report = evaluate(y_arr, mean_arr, std_arr, levels)
        return report, metrics._nll_points(y_arr, mean_arr, std_arr)

    scored = {name: score(pair) for name, pair in ranked_inputs.items()}
    order = sorted(scored, key=lambda name: scored[name][0].nll)  # stable: ties keep given order
    names = tuple(order)
    reports = tuple(scored[name][0] for name in names)

    best: str | None = None
    runner_up: str | None = None
    lead: float | None = None
    lead_se: float | None = None
    if len(names) == 1:
        best = names[0]
    elif len(names) >= 2:
        runner_up = names[1]
        difference = scored[runner_up][1] - scored[names[0]][1]  # positive where the leader wins
        lead = float(np.mean(difference))
        lead_se = float(np.std(difference, ddof=1) / math.sqrt(difference.size))
        if lead > 2.0 * lead_se:
            best = names[0]

    epistemic_scored = {name: score(pair) for name, pair in epistemic_inputs.items()}
    epistemic_names = tuple(epistemic_scored)
    epistemic_reports = tuple(epistemic_scored[name][0] for name in epistemic_names)

    every_report = reports + epistemic_reports  # non-empty: at least one model was given
    return ModelComparison(
        n=every_report[0].n,
        names=names,
        reports=reports,
        best=best,
        runner_up=runner_up,
        lead=lead,
        lead_se=lead_se,
        epistemic_only_names=epistemic_names,
        epistemic_only_reports=epistemic_reports,
        headline=_headline(names, best, lead, lead_se),
    )
