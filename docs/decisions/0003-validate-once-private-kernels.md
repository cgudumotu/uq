# ADR-0003: Validate once at the public boundary; metrics live in private kernels

**Status:** Accepted
**Date:** 2026-09-30
**Deciders:** Carol (line-by-line review); drafted by Claude
**Rules:** API-2, API-4, MATH-3 to MATH-7; TECH-STACK §3

## Context

Every public function must validate its inputs and explain failures (API-2). `evaluate` computes
four metrics at four levels on the same three arrays. If each public metric validated on its own
and `evaluate` called the public metrics, a report on 10⁶ points would re-check the same arrays
sixteen times, and the error messages would be built in several places.

## Decision

Each metric exists once, as a private kernel in `metrics.py` that trusts its inputs
(`_bounds`, `_inside`, `_nll`, `_interval_score`, `_mean_width`). Public functions validate
through `_validate.py`, then call one kernel. `report.evaluate` validates once, then calls kernels
directly. Kernels are never exported and never called from outside `src/uqcalibrate/`.

## Options considered

### Option A: validate in every public function, compose public functions (naive)
| Dimension | Assessment |
|---|---|
| Complexity | Low |
| Cost | O(n) re-validation per metric per level |
| Maintainability | messages duplicated where composition happens |

**Pros:** nothing to explain. **Cons:** 16× validation in `evaluate`; two code paths for the same
metric if `evaluate` is later optimized.

### Option B: validate once at the boundary; private kernels (chosen)
| Dimension | Assessment |
|---|---|
| Complexity | Low–medium: one naming convention to respect |
| Cost | one validation per call |
| Maintainability | one implementation per metric, one message per failure |

**Pros:** single source of truth for each formula and each message; `evaluate` is a plain
orchestration. **Cons:** a kernel called without validation would compute on bad input; the
convention must be enforced.

### Option C: a validated array type (wrapper class) passed between functions
**Pros:** the type system enforces validation. **Cons:** a new public concept for users to learn,
against G1 ("three arrays, one call"); over-engineering at this size.

## Trade-off analysis

B costs one convention; A costs speed and duplication; C costs API surface. The convention is
cheap to enforce: kernels start with an underscore, live only in `metrics.py`, and the API-2 test
suite passes bad input to every public function.

## Consequences

- Easier: adding a level or a metric touches one kernel and one line in `evaluate`.
- Harder: contributors must not expose a kernel; TECH-STACK §3 states the rule.
- Revisit: if a public function ever needs to skip validation for speed, that is a new ADR, not
  a shortcut.

## Action items
1. [x] TECH-STACK §3 amendment: the underscore convention and the kernel rule.
2. [ ] API-2 tests pass NaN, wrong shapes and non-positive `std` to every public function.
