# ADR-0004: Normal quantiles from the standard library; Wilson intervals for verdicts

**Status:** Accepted
**Date:** 2026-09-30
**Deciders:** Carol (P-03 decided the Wilson rule; this ADR records how it is computed)
**Rules:** MATH-2, MATH-7, API-5

## Context

MATH-2 needs Φ⁻¹, the standard normal quantile, for every band. MATH-7 needs a confidence
interval for a binomial proportion (k inside out of n) that behaves at k = 0 and k = n. API-5
allows NumPy and the standard library only.

## Decision

- `z_value(level)` uses `statistics.NormalDist().inv_cdf((1 + level) / 2)`.
- `wilson_interval(k, n, z)` implements the Wilson score interval with z = z_value(0.95), fixed.

## Options considered

### Option A: SciPy (`scipy.stats.norm.ppf`, `scipy.stats.binomtest(...).proportion_ci`)
| Dimension | Assessment |
|---|---|
| Complexity | Low |
| Cost | a 30 MB dependency for two functions; violates API-5 |
| Accuracy | excellent |

### Option B: standard library `NormalDist` + a 10-line Wilson formula (chosen)
| Dimension | Assessment |
|---|---|
| Complexity | Low |
| Cost | zero dependencies |
| Accuracy | `inv_cdf` uses a rational approximation refined to about 1e-15; the MATH-2 oracles are checked to 1e-12 |

### Option C: hard-coded z-values for the four default levels
**Cons:** breaks for any other level; a table is a silent source of typos.

### Interval choices for MATH-7
| Interval | At k = 0 or k = n | Width | Note |
|---|---|---|---|
| normal approximation (Wald) | collapses to zero width | shortest | rejected: gives false certainty at the edges |
| Clopper–Pearson (exact) | fine | widest | conservative; also needs the beta quantile (SciPy) |
| Wilson score (chosen) | fine | between | closed form from z alone; good average coverage (Brown, Cai & DasGupta, 2001) |

## Trade-off analysis

B satisfies API-5 at no accuracy cost that the tests can detect. Wilson is the only interval that
is both closed-form from z (no SciPy) and well-behaved at the edges.

## Consequences

- Easier: no optional dependency to document; the same numbers on every platform.
- Harder: nothing beyond keeping the Wilson formula under test (known answers 183/200 and
  186/200 in MATH-7).
- Revisit: if MATH-7 ever gains a configurable confidence, `wilson_interval` already takes z.

## Action items
1. [ ] MATH-2 oracle test to 1e-12 for the four default levels.
2. [ ] MATH-7 known-answer tests for Wilson at k = 183 and 186 of n = 200, and at k = 0 and k = n.
