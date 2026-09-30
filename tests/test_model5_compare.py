"""MODEL-5a (P-09), MODEL-3 and MATH-4: `compare` ranks models by NLL, shows coverage and width
beside it, says "too close to call" when the lead is within two standard errors, and lists
epistemic-only models separately, never as best.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import uqcalibrate as uqc
from conftest import HALF_LOG_2PI, honest_predictions

NAMES = ("simple neural network", "deep ensemble neural network", "deep operator neural network")


def three_models(rng, n=3_000):
    """One honest model, one overconfident (half std) and one underconfident (double std)."""
    y, mean, std = honest_predictions(rng, n)
    return y, {
        "overconfident": (mean, 0.5 * std),
        "honest": (mean, std),
        "underconfident": (mean, 2.0 * std),
    }


@pytest.mark.rule("MODEL-5")
def test_model5a_orders_by_nll_after_scaling(rng):
    y, predictions = three_models(rng)
    result = uqc.compare(y, predictions)
    assert result.names[0] == "honest"
    nlls = [report.nll for report in result.reports]
    assert nlls == sorted(nlls)
    assert result.best == "honest"
    assert result.runner_up == result.names[1]
    assert result.lead > 2.0 * result.lead_se
    assert result.headline.startswith("Best: honest (leads the runner-up by ")


@pytest.mark.rule("MATH-4")
def test_model5a_never_ranks_by_coverage_alone(rng):
    # sigma x 1000 covers every point and must still come last.
    y, mean, std = honest_predictions(rng, 2_000)
    result = uqc.compare(y, {"honest": (mean, std), "bloated": (mean, 1000.0 * std)})
    assert result.reports[result.names.index("bloated")].delivered[-1] == 1.0
    assert result.names == ("honest", "bloated")
    assert result.best == "honest"


@pytest.mark.rule("MODEL-5")
def test_model5a_identical_models_are_too_close_to_call(honest):
    y, mean, std = honest
    result = uqc.compare(y, {"a": (mean, std), "b": (mean, std.copy())})
    assert result.best is None
    assert result.runner_up == "b"
    assert result.lead == 0.0
    assert (
        result.headline == "Too close to call: the top two differ by 0.00 +/- 0.00 NLL per point."
    )


@pytest.mark.rule("MODEL-5")
def test_model5a_paired_lead_known_answer():
    # Model a: std 1, model b: std 2, both at the truth. Per point, b's NLL exceeds a's by
    # exactly log(2) at every point, so the lead is log(2) with zero standard error.
    y = np.zeros(5)
    result = uqc.compare(y, {"b": (y, np.full(5, 2.0)), "a": (y, np.ones(5))})
    assert result.names == ("a", "b")
    assert result.lead == pytest.approx(np.log(2.0), rel=1e-12)
    assert result.lead_se == 0.0
    assert result.best == "a"
    assert result.reports[0].nll == pytest.approx(HALF_LOG_2PI, rel=1e-12)


@pytest.mark.rule("MODEL-3")
def test_model5a_epistemic_only_listed_separately_and_never_best(rng):
    # An epistemic-only model with a better NLL than every ranked model is still not "best".
    y, mean, std = honest_predictions(rng, 2_000)
    result = uqc.compare(
        y,
        {"ranked": (mean, 0.5 * std)},
        epistemic_only={"dropout without a noise term": (mean, std)},
    )
    assert result.names == ("ranked",)
    assert result.best == "ranked"
    assert result.headline == "Best: ranked (the only model ranked)."
    assert result.epistemic_only_names == ("dropout without a noise term",)
    assert result.epistemic_only_reports[0].nll < result.reports[0].nll
    text = str(result)
    assert "Epistemic only (bands hold no noise term; not ranked)" in text


@pytest.mark.rule("MODEL-3")
def test_model5a_only_epistemic_only_models(honest):
    y, mean, std = honest
    result = uqc.compare(y, {}, epistemic_only={"x": (mean, std)})
    assert result.names == ()
    assert result.best is None
    assert result.headline == "No model reports total uncertainty; nothing ranked."


@pytest.mark.rule("MODEL-5")
def test_model5a_table_text(rng):
    y, mean, std = honest_predictions(rng, 60)
    result = uqc.compare(
        y,
        {NAMES[1]: (mean, std), NAMES[2]: (mean, 0.9 * std), NAMES[0]: (mean, 0.5 * std)},
        epistemic_only={"Monte Carlo dropout neural network": (mean, 0.4 * std)},
    )
    text = str(result)
    assert text.isascii()
    lines = text.splitlines()
    assert all(len(line) <= 100 for line in lines)
    assert lines[0] == "uqcalibrate: 3 models ranked, 1 epistemic-only, 60 points"
    assert lines[1].startswith("rank  model")
    for column in ("95% delivered", "mean width", "NLL", "verdict"):
        assert column in lines[1]
    assert lines[2].startswith("   1  ")
    assert lines[3].startswith("   2  ")
    assert lines[4].startswith("   3  ")
    assert lines[5] == result.headline
    assert lines[6] == ""
    assert lines[7] == "Epistemic only (bands hold no noise term; not ranked)"
    assert lines[8].startswith("   -  Monte Carlo dropout neural network")


@pytest.mark.rule("MODEL-5")
def test_model5a_to_dict_is_plain_python(rng):
    y, predictions = three_models(rng, n=200)
    d = uqc.compare(y, predictions).to_dict()
    json.dumps(d)
    assert set(d) == {
        "n",
        "names",
        "reports",
        "best",
        "runner_up",
        "lead",
        "lead_se",
        "epistemic_only_names",
        "epistemic_only_reports",
        "headline",
    }
    assert d["reports"][0]["headline"].startswith(("CONSISTENT", "OVERCONFIDENT", "UNDERCONFIDENT"))


@pytest.mark.rule("MODEL-5")
def test_model5a_is_immutable_and_pure(rng, capsys):
    y, predictions = three_models(rng, n=200)
    first = uqc.compare(y, predictions)
    second = uqc.compare(y, predictions)
    assert first == second
    assert capsys.readouterr() == ("", "")
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        first.best = "x"  # type: ignore[misc]


@pytest.mark.rule("API-2")
def test_model5a_validation_messages(honest):
    y, mean, std = honest
    with pytest.raises(
        ValueError, match=r"^predictions must be a mapping from model name to \(mean, std\)"
    ):
        uqc.compare(y, [(mean, std)])
    with pytest.raises(
        ValueError, match=r"^predictions\['a'\] must be a pair \(mean, std\) \(got 3 items\)"
    ):
        uqc.compare(y, {"a": (mean, std, std)})
    with pytest.raises(ValueError, match=r"^compare needs at least one model"):
        uqc.compare(y, {})
    with pytest.raises(ValueError, match=r"^model names must be non-empty strings \(got 3\)"):
        uqc.compare(y, {3: (mean, std)})
    with pytest.raises(ValueError, match=r"^'a' appears in both predictions and epistemic_only"):
        uqc.compare(y, {"a": (mean, std)}, epistemic_only={"a": (mean, std)})
    with pytest.raises(ValueError, match=r"^compare needs at least 2 points"):
        uqc.compare(y[:1], {"a": (mean[:1], std[:1])})
    with pytest.raises(ValueError, match=r"^std has 1 value <= 0"):
        uqc.compare(y, {"a": (mean, np.where(np.arange(len(std)) == 0, 0.0, std))})
