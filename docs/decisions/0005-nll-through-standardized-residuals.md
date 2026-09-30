# ADR-0005: The Gaussian NLL is computed through standardized residuals

**Status:** Accepted
**Date:** 2026-09-30
**Deciders:** Carol (review); drafted by Claude
**Rules:** MATH-5, API-2

## Context

MATH-5 defines `NLL = mean(½·log(2π·σ²) + (y − μ)² / (2·σ²))`. Coded literally, `σ²` underflows
to 0.0 for σ below about 1e-154, which turns `log(σ²)` into `-inf` and the division into a
warning, even though API-2 has admitted the input (σ > 0). The design claim "no floating-point
warnings on validated input" would be false.

## Decision

Compute the same quantity as `mean(½·log(2π) + log(σ) + ½·z²)` with `z = (y − μ) / σ`. This is
algebraically identical (`½·log(σ²) = log(σ)`; `(y − μ)² / (2σ²) = ½·z²`) and never squares σ.
The constant `½·log(2π) = 0.9189385332046727` is written once, in `metrics.py`.

## Options considered

### Option A: literal formula
**Pros:** matches the rule text symbol for symbol. **Cons:** underflow at extreme σ; a squared
temporary array.

### Option B: standardized-residual form (chosen)
**Pros:** no underflow until σ itself underflows (which API-2's finiteness check would not
admit as 0 anyway); one fewer temporary; the same `z` array serves the MATH-8 formula later.
**Cons:** a reviewer must do one line of algebra to see it equals the rule.

## Consequences

- The MATH-5 known answer (y = μ, σ = 1 gives 0.9189385332046727) is unchanged.
- A test asserts the two forms agree to 1e-12 on random inputs, so the algebra is checked by
  machine, not only by eye.
- `interval_score` stays in y-units (widths and misses), as MATH-6 defines it.

## Action items
1. [ ] `test_math5_two_forms_agree` on seeded inputs.
