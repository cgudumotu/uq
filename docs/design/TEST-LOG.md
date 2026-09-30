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
