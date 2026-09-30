"""The example outputs printed in README.md are real: regenerate them and compare verbatim.

If a change to the report or the comparison table alters the output, these tests fail until the
README (and MISSION section 6, which shows the same blocks) is updated, so the documentation can
never drift from the code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import uqcalibrate as uqc
from conftest import honest_predictions

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", ROOT / "docs" / "constitution" / "MISSION.md"]


def documents() -> list[str]:
    found = [path.read_text(encoding="utf-8") for path in DOCS if path.exists()]
    if not found:
        pytest.skip("README.md is not present next to the tests")
    return found


@pytest.mark.rule("MATH-7")
def test_math7_readme_report_is_real():
    # README: "200 points from a seeded example whose model reports half the true spread".
    y, mean, std = honest_predictions(np.random.default_rng(1), 200, scale=0.5)
    text = str(uqc.evaluate(y, mean, std))
    for document in documents():
        assert text in document


@pytest.mark.rule("MODEL-5")
def test_model5a_readme_comparison_is_real():
    # README: 60 seeded points; models at 0.5, 0.9 and 1.0 times the true spread; one
    # epistemic-only model at 0.35.
    y, mean, std = honest_predictions(np.random.default_rng(3), 60)
    result = uqc.compare(
        y,
        {
            "simple neural network": (mean, 0.5 * std),
            "deep ensemble neural network": (mean, std),
            "deep operator neural network": (mean, 0.9 * std),
        },
        epistemic_only={"Monte Carlo dropout neural network": (mean, 0.35 * std)},
    )
    text = str(result)
    for document in documents():
        assert text in document
