# Code review: uqcalibrate M1 core (2026-09-30)

**Scope:** `src/uqcalibrate/` (seven modules, 226 statements) and `tests/` (88 tests), reviewed
against MISSION MATH-1..8 and API-1..6, TECH-STACK import rules 1–5, ADR-0003..0005 and the SOLID
principles. Security dimensions (injection, secrets, deserialization) do not apply: the package
does no I/O and evaluates no input.

## Summary

Every formula matches its rule text line for line, every error message matches UX-COPY.md, and
the suite reaches 100% line coverage. The review found no correctness defect in the math. It
found three maintainability items, all fixed before this record was written.

## Findings (all fixed)

| # | File | Issue | Category | Fix |
|---|---|---|---|---|
| 1 | `_validate.py` | `_as_1d` special-cased its caller by checking `name == "mean"`, so a generic helper knew who was calling it. | Single responsibility | Split: `_as_1d` handles the column-vector mistake for any argument; a new `_mean_1d` owns the "samples passed as a mean" message. |
| 2 | `_validate.py` | `levels=np.array(0.95)` (a 0-d array) reached `len()` and leaked a `TypeError`, against API-2, which promises `ValueError`. | Correctness (error type) | A 0-d array is now rejected as "not a sequence" with the UX-COPY message; test added. |
| 3 | `report.py` | `evaluate` annotated `levels` as `tuple[float, ...]` while accepting any sequence. | Documentation honesty | Annotated `Sequence[float]`. |

Also considered and left as is:

- `_as_1d(std_raw, "std")` converts an array that is already float64; `np.asarray` returns it
  unchanged, so there is no copy.
- `HEADLINE_LEVEL not in level_tuple` uses exact float equality. A user who writes `0.95`
  matches; one who computes `0.9500000001` gets the MATH-7 `NotImplementedError`, whose message
  says to add `0.95`. Acceptable until HANDOFF Q-1 is decided.
- `_WILSON_Z` is computed at import time. It is pure and costs microseconds.

## Rule-by-rule check

| Rule | Where | Verified by reading | Verified by test |
|---|---|---|---|
| MATH-1 | `combine.total_std` | `sqrt(noise_vars.mean(0) + means.var(0))`; NumPy `var` defaults to `ddof=0` (P-01) | 3-4-5, two-member, S = 1, divisor, thesis formula rejected |
| MATH-2 | `_normal.z_value`, `metrics._bounds` | `inv_cdf((1 + level) / 2)`; band `mean ± z·std` | four z oracles to 1e-12; 1.97σ point is outside; level range |
| MATH-3 | `metrics._inside`, `coverage` | `(lower <= y) & (y <= upper)`, inclusive | boundary, all in/out, oracle ±0.005, scalar std |
| MATH-4 | `report.evaluate` | coverage, width, interval score per level; NLL once | fields per level; inflated σ loses width and NLL |
| MATH-5 | `metrics._nll` | `½·log 2π + log σ + ½·z²` (ADR-0005) | constant; residual known answer; two forms agree; minimum at true σ; no clamp |
| MATH-6 | `metrics._interval_score` | `width + (2/α)·(below + above)` | inside = width; known miss above and below; minimum at honest σ |
| MATH-7 | `_normal.wilson_interval`, `report._verdict` | Wilson formula; `high < p` / `low > p` / else | four Wilson oracles; 183 and 186 of 200; oracle at 10⁵; halved σ; n = 10; missing 0.95 raises |
| MATH-8 | `calibration.fit_scaling` | validate → zero-residual `ValueError` → `NotImplementedError` naming MATH-8/P-04 | both errors; acceptance test `xfail(strict=True)` |
| API-1 | `__init__.__all__` | seven names; no `__future__` leak | `dir()` minus modules == the seven |
| API-2 | `_validate` | every message in UX-COPY.md | one test per message; NaN rejected by every public function; 100% of validator branches |
| API-3 | signatures | `std`, `noise_vars`, `means`, `mean_widths` | signature inspection |
| API-4 | all | no print, no RNG, frozen dataclass, inputs untouched | stdout empty; RNG state unchanged; `FrozenInstanceError`; inputs equal after call |
| API-5 | imports | NumPy + stdlib; no `models`, `torch`, `reproduce`, `tests` | subprocess import with `torch` blocked; AST scan |
| API-6 | all text in `src/` | full model names only | regex scan for abbreviations |

## What looks good

- One kernel per metric, one validator, one message per failure (ADR-0003): a reviewer can
  point at the single line that implements each rule.
- The tests were seen failing first (TEST-LOG.md) and the `xfail(strict=True)` on MATH-8 makes
  the pending decision impossible to forget: the day the formula lands, the suite fails until
  the marker is removed.
- The report renders identically on every platform: ASCII only, 79 columns, verified by test.
- The docstring examples run as tests (`--doctest-modules`), so the documentation cannot drift.

## Addendum: P-04 and P-09 (2026-09-30, later the same day)

- `calibration.py`: the pending line became `float(np.sqrt(np.mean(residuals * residuals)))`.
  Checked against MATH-8 symbol by symbol; the all-zero refusal precedes it, so a division by a
  zero mean cannot occur. Known answers 2 and √12.5, the unit-RMS property and the "no other
  factor scores lower" property are tested.
- `metrics.py`: `_nll` now averages a new per-point kernel `_nll_points`; the MATH-5 tests
  (constant, residual known answer, two forms agree) are unchanged and still pass.
- `compare.py`: reviewed for the rule text of MODEL-5a. Sorting is Python's stable sort on
  `report.nll`, so ties keep the caller's order. The paired standard error uses `ddof=1` over
  n ≥ 2 points (validated). `best` is set only when `lead > 2 * lead_se`, so a lead of exactly
  zero with zero standard error is "too close to call" (two identical models). Epistemic-only
  models are scored with the same `evaluate` but never enter `names`, so they cannot become
  `best`. One simplification made during review: `n` is read from the first report instead of
  being recomputed from the arrays.
- Validation messages follow the UX-COPY addendum; one test per message. 100% coverage holds
  (326 statements).

## Verdict

**Approve for Carol's line-by-line review.** Nothing is committed; Carol commits from Cursor.
