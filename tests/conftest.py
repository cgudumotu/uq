"""Shared fixtures. Every test names its MISSION rule with @pytest.mark.rule (TECH-STACK section 5).

Nothing here depends on global random state (DATA-5): every random draw comes from a seeded
generator handed out by a fixture.
"""

from __future__ import annotations

import numpy as np
import pytest

# Oracles for the tests, written here once (HANDOFF "Constants").
Z_95 = 1.9599639845400536
Z_90 = 1.6448536269514715
Z_80 = 1.2815515655446008
Z_50 = 0.6744897501960817
HALF_LOG_2PI = 0.9189385332046727
DEFAULT_LEVELS = (0.5, 0.8, 0.9, 0.95)


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded generator. Tests that need a different seed ask for default_rng themselves."""
    return np.random.default_rng(20260930)


def honest_predictions(rng: np.random.Generator, n: int, scale: float = 1.0):
    """Data whose bands are exactly honest: y ~ N(mean, std**2).

    `scale` multiplies the *reported* std only, so scale=0.5 is a model that is overconfident by
    a factor of two while the data stay the same.
    """
    mean = rng.normal(0.0, 3.0, size=n)
    std = rng.uniform(0.5, 2.0, size=n)
    y = mean + std * rng.standard_normal(n)
    return y, mean, scale * std


@pytest.fixture
def honest(rng):
    """(y, mean, std) for 2,000 honest points."""
    return honest_predictions(rng, 2_000)
