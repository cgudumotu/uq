# TECH-STACK — uqcalibrate

**Status:** DRAFT constitution v1 · 2026-09-30 · amended the same day (§10) · awaiting Carol's approval
**Scope:** How MISSION.md is built. A change to this document is recorded as an ADR in
docs/decisions/ and appended to §10 with a date and a reason.

---

## 1. Language and platforms

- **Python ≥ 3.10.** CI tests 3.10 through 3.14. (Python 3.10 reaches end of life in October
  2026; dropping it later is a minor-version decision recorded as an ADR.)
- **CPU only.** Nothing requires a GPU, including reproduce/.
- **Windows, macOS and Linux.** Windows matters: it is the maintainer's platform.
- **float64 everywhere** in the core.

## 2. Tooling

| Job | Tool | Command |
|---|---|---|
| Environments, lockfile, running | uv | `uv sync`, `uv run …` |
| Build backend | hatchling (PEP 517) | `uv build` |
| Tests | pytest, pytest-cov | `uv run pytest` |
| Lint and format | ruff | `uv run ruff check .`, `uv run ruff format .` |

`uv.lock` is committed so every machine resolves the same versions.

## 3. Layout, module boundaries and import rules

```
uq/
  src/uqcalibrate/
    __init__.py       re-exports the public API (MISSION API-1) and nothing else
    _validate.py      input checks shared by every module (API-2)
    _normal.py        z-values from statistics.NormalDist (MATH-2)
    combine.py        total_std: the only place total variance is computed (MATH-1)
    metrics.py        coverage, gaussian_nll, interval_score (MATH-3, 5, 6)
    calibration.py    fit_scaling (MATH-8)
    compare.py        compare and ModelComparison: ranks models (MODEL-5a, MODEL-3)
    report.py         evaluate and CalibrationReport: composes metrics, explains (MATH-4, 7)
    models/           M2 only, PyTorch extra (MODEL-1..4)
  tests/
  reproduce/          published DONN computation + regeneration command (REPRO-1..5); not packaged
  legacy/notebooks/   stripped copies of the original notebooks (REPRO-5)
  docs/constitution/  MISSION.md, TECH-STACK.md, ROADMAP.md
  docs/decisions/     ADRs
  docs/design/        Phase 1 outputs
```

**Single responsibility.** Models predict; `combine` combines uncertainty; `metrics` scores;
`report` explains; `calibration` fixes.

**Dependency direction (arrows point at what a module may import; amended 2026-09-30, ADR-0003):**

```
__init__ ─► report ──────► metrics ─► _normal
    │          ├─────────► _normal        │
    │          └─────────► _validate ◄────┘
    ├──────► compare ─────► report, metrics, _validate
    ├──────► metrics   (public wrappers)
    ├──────► combine ─────► _validate
    └──────► calibration ─► _validate
_normal, _validate ─► NumPy and the standard library only
models (M2) ─► core modules         core ─X─► models, torch, reproduce, tests
reproduce ─► uqcalibrate            src ─X─► reproduce
```

**Import rules, enforced by tests:**

1. Core modules import only NumPy, the standard library and other core modules (API-5).
2. No module under src/ imports reproduce/ or tests/ (REPRO-3).
3. `uqcalibrate.models` may import the core; the core never imports `models` or `torch`. A test
   imports `uqcalibrate` with `torch` made unimportable and must succeed.
4. A leading underscore means *internal to the package*: sibling modules may use the name, but it
   is never re-exported and never part of the public surface. Only `__init__.py` defines the
   public surface.
5. Metric kernels (`metrics._bounds`, `_inside`, `_nll`, `_interval_score`, `_mean_width`) trust
   their inputs. Every public function validates through `_validate` before calling a kernel;
   `report.evaluate` validates once and then calls kernels directly (ADR-0003). No kernel is
   called from outside `src/uqcalibrate/`.

## 4. Dependencies

| Group | Contents | Installed by |
|---|---|---|
| core | `numpy>=1.24` | `pip install uqcalibrate` |
| extra `torch` (M2) | `torch` | `pip install "uqcalibrate[torch]"` |
| dev | pytest, pytest-cov, ruff | `uv sync` (development only) |
| reproduce | torch, scipy, pandas, matplotlib, seaborn: what Refactor_donn.ipynb imports (ADR-0001) | `uv sync --group reproduce`; repository only, never in the wheel |

No SciPy in the core: the only special function M1 needs is the normal quantile, which the
standard library provides. Exact versions of the reproduce group are pinned once the published
computation runs (REPRO-4).

## 5. Testing standards

- **Tests first.** A test is written, run and seen failing before the code it tests exists.
- **Every test names its rule.** The function name starts with the rule ID
  (`test_math1_three_four_five`), and the test carries `@pytest.mark.rule("MATH-1")`. A meta-test
  reads MISSION.md and fails if a rule in force has no test (MISSION G5).
- **Kinds of test:**
  - *Known-answer:* exact values, `rtol=1e-12` (3-4-5, two-member ensemble, NLL constant).
  - *Property:* invariants (coverage in [0, 1]; the interval score is lowest at the honest σ).
  - *Statistical:* seeded, with the tolerance derived from the binomial or normal standard error
    and written in the test (the oracle reaches nominal coverage).
  - *Golden:* reproduce/ against reproduce/published_values.csv (REPRO-2), marked
    `@pytest.mark.repro`, excluded from the default run.
  - *Boundary:* the import rules in §3.
- **Speed.** `uv run pytest` (the default, fast suite) finishes in under 30 s.
  `uv run pytest -m repro` runs the golden tests.
- **Coverage.** At least 95% line coverage of src/uqcalibrate, reported in CI.
- **Determinism.** No test depends on global random state. A flaky test is a bug.
- **Configuration (pyproject `[tool.pytest.ini_options]`).** The `rule` and `repro` markers are
  registered, so an unregistered marker is an error; `filterwarnings = ["error"]` turns every
  warning into a failure, which enforces the "no floating-point warnings on validated input"
  claim (SYSTEM-DESIGN §3.7); the default `addopts` deselects `repro`.

## 6. Code standards

- Type hints on every public function. NumPy-style docstrings with a runnable example.
- `ruff` rule sets: E, F, W, I (imports), B (bugbear), UP (pyupgrade), NPY (NumPy), PT (pytest).
- No `print` and no logging in the core (API-4).
- Errors follow MISSION API-2: ValueError naming the argument, the problem, the likely cause and
  the fix.

## 7. Repository hygiene

- `.gitignore` excludes AI-tooling files: `CLAUDE.md`, `.claude/`, `.cursor/`, `.cursorrules`,
  plus Python build and cache artifacts, `.venv/`, Jupyter checkpoints and operating-system files
  (`Thumbs.db`, `desktop.ini`, `.DS_Store`). The public repository contains only what a user or
  reviewer needs.
- legacy/notebooks/ copies have outputs stripped and code cells unchanged (REPRO-5b).
- Original evidence files are never edited or moved (see the evidence list in ROADMAP.md, M0).
- The project folder sits inside OneDrive. Git works there, but sync can briefly lock files during
  commits; if a commit fails with a lock error, pause OneDrive sync and retry.

## 8. Release standards

- **Semantic versioning**, starting at 0.1.0; 0.x until the API is declared stable. The version
  is written once, as `__version__` in `src/uqcalibrate/__init__.py`; hatchling reads it from
  there (`[tool.hatch.version] path`), so pyproject.toml carries no second copy.
- `CHANGELOG.md` in Keep a Changelog format.
- **Release checklist:** default and repro suites green; ruff clean; README examples run;
  ROADMAP.md updated.
- **Publishing:** TestPyPI first, then PyPI. **Each upload needs Carol's explicit approval.**
  Publishing will use PyPI Trusted Publishing from GitHub Actions, so no API token is stored (ADR
  in M2).

## 9. Continuous integration

GitHub Actions on every push and pull request: `ruff check`, `ruff format --check`, and the
default pytest suite on Python 3.10–3.14, on ubuntu-latest and windows-latest.

Third-party actions are pinned to exact release versions (for example `actions/checkout@v7.0.1`),
so a new release of an action cannot change the build without a commit in this repository.
`astral-sh/setup-uv` has published no major-version tags since v8, so an exact version is the
only option there.

## 10. Amendment log

- **2026-09-30** v1 drafted. Not yet approved.
- **2026-09-30, amendment 1 (ADR-0001).** reproduce/ keeps the published computation for the DONN
  only, so the reproduce group drops `blitz-bayesian-pytorch`, which only the published BNN used.
  It keeps pandas and seaborn: Refactor_donn.ipynb uses them for its results table and its band
  plot (checked 2026-09-30). §7 also lists Jupyter checkpoints and operating-system files, because
  the project folder sits on a Windows desktop synced by OneDrive, where Windows can create
  `desktop.ini` and `Thumbs.db` on its own.
- **2026-09-30, amendment 2 (Phase 2 step 2, ADR-0003 to ADR-0005).** §3: the dependency
  diagram now matches SYSTEM-DESIGN.md (`report` uses `metrics`, `_normal` and `_validate`, not
  `combine`); import rules 4 and 5 state the underscore convention and the kernel rule. §5: the
  pytest configuration (registered markers, warnings as errors). §8: the version is written once
  in `__init__.py`.
- **2026-09-30, amendment 3 (P-09).** §3: `compare.py` added to the core layout and the
  dependency diagram; it imports `report`, `metrics` and `_validate` and, like every core
  module, nothing outside NumPy and the standard library.
- **2026-09-30, amendment 4.** §9: CI is live; actions pinned to exact versions. The first two
  CI runs passed but warned that `actions/checkout@v4` and `astral-sh/setup-uv@v6` target
  Node.js 20, which GitHub deprecated; they now run `actions/checkout@v7.0.1` and
  `astral-sh/setup-uv@v9.0.0`, both on Node.js 24. No breaking change in setup-uv v7–v9 affects
  this workflow (checked against their release notes).
