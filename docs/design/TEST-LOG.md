# Test log: tests first (TECH-STACK section 5)

## Red run, 2026-09-30

The suite was written before any implementation. Against the empty package
(`src/uqcalibrate/__init__.py` holding only `__version__`):

```
uv run pytest -q --continue-on-collection-errors
51 failed, 5 passed, 2 errors in 0.45s
```

- 2 collection errors: `test_math2_bands.py` and `test_math7_verdict.py` import `uqcalibrate._normal`,
  which did not exist.
- 51 failures: `AttributeError: module 'uqcalibrate' has no attribute ...` for every public
  function.
- 5 passes, all structural and vacuously true on an empty package: the three API-5 import-rule
  tests, the API-6 no-abbreviations scan, and the G5 meta-test (every M1 rule ID already appears
  in a test marker).

## Green run, 2026-09-30

After the implementation, the review fixes (CODE-REVIEW.md) and the extra validator tests:

```
uv run ruff check .                 All checks passed!
uv run ruff format --check .        34 files already formatted
uv run pytest -q                    88 passed, 2 skipped, 1 xfailed in 0.50s
uv run pytest --cov=uqcalibrate     100% line coverage (226 statements)
uv build                            uqcalibrate-0.1.0-py3-none-any.whl, 7 modules + LICENSE
```

- 2 skipped: the two docstring examples marked `+SKIP` (they need real data).
- 1 xfailed: `test_math8_recovers_factor_two_and_fixes_the_report`, strict, until P-04 is decided.
- The wheel installs into a clean environment and imports with `torch` blocked (API-5).

## P-04 and P-09, 2026-09-30 (later the same day)

Tests first again. `test_math8_scaling.py` rewritten for the decided formula: red run
`6 failed, 2 passed` against the pending stub (the two in-force clauses kept passing). New
`test_model5_compare.py` (11 tests) plus the API-1 and API-3 updates: red run
`12 failed, 92 passed` with `AttributeError: module 'uqcalibrate' has no attribute 'compare'`.

Green run after `calibration.py` (one line) and the new `compare.py`:

```
uv run ruff check .                 All checks passed!
uv run pytest -q                    105 passed, 1 skipped in 0.36s
uv run pytest --cov=uqcalibrate     100% line coverage (326 statements)
```

The strict `xfail` on the MATH-8 acceptance test did its job: it started passing, the suite
failed with an unexpected pass, and the marker came off in the same change.
