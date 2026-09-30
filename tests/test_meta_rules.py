"""G5: every rule in force for M1 has at least one test that names it (TECH-STACK section 5)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
MISSION = TESTS.parent / "docs" / "constitution" / "MISSION.md"

# M1 scope (MISSION section 8): every MATH and API rule. DATA-1, DATA-2 and DATA-5 join on
# 2026-10-03 with reproduce/data.py (HANDOFF "Acceptance tests"); REPRO and MODEL rules are
# tested by reproduce/ (M1, Oct 3) and models/ (M2).
M1_RULES = {f"MATH-{i}" for i in range(1, 9)} | {f"API-{i}" for i in range(1, 7)}
MARKER = re.compile(r"""pytest\.mark\.rule\(\s*["']([A-Z]+-\d+)["']\s*\)""")
HEADING = re.compile(r"\*\*((?:MATH|DATA|REPRO|API|MODEL)-\d+)\b")


def rules_named_in_tests() -> set[str]:
    found: set[str] = set()
    for path in TESTS.glob("test_*.py"):
        found |= set(MARKER.findall(path.read_text(encoding="utf-8")))
    return found


@pytest.mark.rule("API-1")  # the meta-test itself is anchored to the rule that fixes the surface
def test_meta_every_in_force_rule_has_a_test():
    if not MISSION.exists():
        pytest.skip("MISSION.md is not shipped in the sdist; run from the repository")
    in_mission = set(HEADING.findall(MISSION.read_text(encoding="utf-8")))
    assert M1_RULES <= in_mission, "MISSION.md lost a rule heading"
    named = rules_named_in_tests()
    missing = sorted(M1_RULES - named)
    assert not missing, f"rules in force without a test: {missing}"
    unknown = sorted(named - in_mission)
    assert not unknown, f"tests name rules that MISSION.md does not define: {unknown}"
