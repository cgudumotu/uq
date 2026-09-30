# System design: uqcalibrate M1

**Phase 2, step 1 · 2026-09-30.** Inputs: MISSION.md (rules), TECH-STACK.md (tooling, layout),
HANDOFF.md (migration map, edge cases, tests). Output: the module design that the tests and code
follow. **Gates:** this document adds no public name, signature, default or error type beyond
MISSION §6 as amended on 2026-09-30, so no approval gate is reached. Two internal decisions are
recorded as ADRs (0003, 0004) in the architecture step.

## 1. Requirements

**Functional (from MISSION).**
- F1 Combine sampled predictions into one σ per point by the law of total variance (MATH-1).
- F2 Build central bands at any level from the normal quantile (MATH-2); never ±2σ.
- F3 Score a band: coverage (MATH-3), Gaussian NLL (MATH-5), interval score (MATH-6).
- F4 Produce a report that always shows coverage, width and a proper score together (MATH-4),
  with a Wilson-interval verdict per level and a headline at 95% (MATH-7).
- F5 Fit one scaling factor on held-out data (MATH-8; objective pending P-04).
- F6 Validate every input and explain every failure (API-2); keep the six names (API-1); keep
  names honest about units (API-3); stay pure (API-4); depend on NumPy only (API-5); print full
  model names (API-6).

**Non-functional.**
- N1 Correctness first: known answers to 1e-12; statistical tests with derived tolerances.
- N2 Size: n up to about 10⁷ points in memory (a handful of float64 temporaries per call);
  runtime O(n × levels); no Python loops over points.
- N3 Determinism: same inputs, same outputs, on every platform; float64 throughout.
- N4 Portability: Python 3.10–3.14, Windows/macOS/Linux, CPU only; report output ASCII.
- N5 Test suite under 30 s; import time negligible (no heavy imports at module load).

**Constraints.** One developer-reviewer pair; every line reviewed; M1 by 2026-10-03; the
published computation lives only in reproduce/ (REPRO-3); no PyTorch in the core.

## 2. High-level design

### 2.1 Components

```
 user code                       uqcalibrate (src/)                              policy source
 ─────────                       ──────────────────                              ─────────────
 y, mean, std ──────────────►  __init__.py  (public surface, API-1)
                                   │
                    ┌──────────────┼───────────────────┬─────────────────┐
                    ▼              ▼                   ▼                 ▼
               combine.py      metrics.py          report.py       calibration.py
               total_std       coverage            evaluate         fit_scaling
               (MATH-1)        gaussian_nll        CalibrationReport (MATH-8)
                               interval_score      (MATH-4, MATH-7)
                               (MATH-2,3,5,6)        │   │
                    │              │   │              │   │                │
                    │              │   ▼              │   ▼                │
                    │              │ _normal.py ◄─────┘  (z_value,         │
                    │              │ (MATH-2 quantile,    wilson_interval)  │
                    │              │  Wilson math)                          │
                    ▼              ▼                  ▼                     ▼
                              _validate.py  (API-2: one place for every check and message)
```

Arrows point at what a module imports. `_normal.py` and `_validate.py` import only NumPy and
the standard library. No module under `src/` imports `models`, `torch`, `reproduce` or `tests`.

### 2.2 Data flow: from three arrays to a report

```
(y, mean, std, levels)
   │
   ▼  _validate.prediction_inputs()            float64, shape (n,), finite, std > 0, n >= 1
   │  _validate.levels()                       each in (0, 1), non-empty, no duplicates
   │  0.95 in levels?  no ──► NotImplementedError (MATH-7, HANDOFF Q-1)
   ▼
for each level p:
   z      = _normal.z_value(p)                 Φ⁻¹((1 + p) / 2)                      MATH-2
   lower  = mean − z·std ;  upper = mean + z·std                                      MATH-2
   inside = (lower <= y) & (y <= upper)         inclusive                             MATH-3
   k      = inside.sum() ; delivered = k / n                                          MATH-3
   low, high = _normal.wilson_interval(k, n, z95)                                     MATH-7
   verdict  = OVERCONFIDENT if high < p ; UNDERCONFIDENT if low > p ; else CONSISTENT MATH-7
   width    = mean(2·z·std)                                                           MATH-4
   score    = mean(width_i + (2/α)·misses)                                            MATH-6
once:
   nll = mean(½·log(2π·std²) + (y − mean)² / (2·std²))                                MATH-5
   headline = verdict at 0.95, formatted per UX-COPY                                  MATH-7
   ▼
CalibrationReport(n, levels, delivered, plausible_ranges, verdicts, mean_widths,
                  interval_scores, nll, headline)   frozen                             §6
   │
   ├── str(report)   → fixed-width ASCII table + footer + headline (UX-COPY.md)
   └── to_dict()     → plain Python types
```

`total_std` is a separate, shorter flow: validate `(S, n)` pair → `sqrt(noise_vars.mean(0) +
means.var(0))` (population variance, P-01) → `(n,)`.

### 2.3 API contracts

Exactly MISSION §6, as amended 2026-09-30 (amendment 7). Repeated here only to fix the types:

| Name | In | Out | Raises |
|---|---|---|---|
| `total_std(means, noise_vars)` | array-likes `(S, n)`, S ≥ 1, `noise_vars` ≥ 0 | `ndarray[float64] (n,)` | `ValueError` |
| `coverage(y, mean, std, level=0.95)` | `(n,)`, `(n,)`, `(n,)` or scalar; `0 < level < 1` | `float` | `ValueError` |
| `gaussian_nll(y, mean, std)` | as above | `float` | `ValueError` |
| `interval_score(y, mean, std, level=0.95)` | as above | `float` | `ValueError` |
| `evaluate(y, mean, std, levels=(0.5, 0.8, 0.9, 0.95))` | as above; `levels` a sequence | `CalibrationReport` | `ValueError`, `NotImplementedError` (0.95 absent) |
| `fit_scaling(y, mean, std)` | as above | `float > 0` | `ValueError` (all residuals zero), `NotImplementedError` (P-04) |

Return scalars are Python `float`, not NumPy scalars, so they print and serialize plainly.

### 2.4 Storage

None in the package (API-4). `reproduce/` writes files under `reproduce/output/` only (§5).

## 3. Deep dive

### 3.1 Validation layer (`_validate.py`)

One module owns every check and every message, so a message is written once and tested once.

| Function | Checks | Returns |
|---|---|---|
| `prediction_inputs(y, mean, std)` | `y`, `mean` → float64 1-D of equal length n ≥ 1; `std` → float64 `(n,)` (scalar broadcast); all finite; `std > 0`. Detects `mean.ndim == 2` and uses the "samples passed as a mean" message. | `(y, mean, std)` |
| `sample_inputs(means, noise_vars)` | both float64 2-D, same shape `(S, n)`, S ≥ 1, n ≥ 1, finite; `noise_vars >= 0`. 1-D input gets the "use `means[None, :]`" message (HANDOFF D-1). | `(means, noise_vars)` |
| `level(p)` | `0 < p < 1`; the message shows `level=0.95` as the example. | `float` |
| `levels(seq)` | non-empty; each passes `level`; no duplicates (HANDOFF D-2). | `tuple[float, ...]` |

Message construction follows UX-COPY.md: the count and the extreme value (`3 values <= 0
(min -0.2)`), the requirement, the fix as code. The first offending index is reported for
non-finite values. Messages are plain `str`; no f-string is built outside this module.

### 3.2 Normal-distribution math (`_normal.py`)

- `z_value(level) -> float`: `statistics.NormalDist().inv_cdf((1 + level) / 2)`. Assumes a
  validated level; documented as internal.
- `wilson_interval(k, n, z) -> tuple[float, float]`: center `(p̂ + z²/2n) / (1 + z²/n)`, half-width
  `z·√(p̂(1−p̂)/n + z²/4n²) / (1 + z²/n)`, clipped to [0, 1]. Defined for k = 0 and k = n, which
  is the reason for choosing Wilson over the normal approximation. Pure math, no policy: the
  verdict rule lives in `report.py`.

### 3.3 Metrics: one kernel per metric, validate once (`metrics.py`)

Each metric exists once, as a private kernel that trusts its inputs:

```
_bounds(mean, std, z)                     -> (lower, upper)
_inside(y, lower, upper)                  -> bool array (n,)     coverage = _inside(...).mean()
_nll(y, mean, std)                        -> float               ADR-0005 form
_interval_score(y, lower, upper, alpha)   -> float
_mean_width(std, z)                       -> float
```

`_inside` returns the mask rather than the fraction because `evaluate` needs the count k for
the Wilson interval and the fraction for the table; one kernel serves both.

The public functions validate through `_validate`, then call one kernel. `report.evaluate`
validates once and calls the kernels directly for every level, so a report on 10⁶ points does not
re-check the same arrays four times. Kernels are never exported (ADR-0003).

### 3.4 Report (`report.py`)

- `CalibrationReport`: `@dataclass(frozen=True)`; tuple fields, so the object is hashable and
  cannot be edited after the fact (API-4). `__str__` renders per UX-COPY.md with fixed column
  widths; `to_dict()` returns `dict[str, int | float | str | list]`, with `plausible_ranges` as a
  list of two-element lists.
- `evaluate`: the flow in §2.2. The `NotImplementedError` for a missing 0.95 is raised before any
  metric is computed, with the UX-COPY message.
- Verdict strings are module-level constants (`OVERCONFIDENT = "OVERCONFIDENT"`, …) so tests and
  M2 code compare against names, not literals.

### 3.5 Calibration (`calibration.py`)

`fit_scaling` validates, computes the standardized residuals `(y − mean) / std`, raises
`ValueError` if all are zero (E-d), then raises `NotImplementedError` naming MATH-8 and P-04. When
P-04 is decided, the pending line becomes the one-line formula and the `xfail` test flips to a
passing test (HANDOFF D-4).

### 3.6 Public surface (`__init__.py`)

Imports the six functions and `CalibrationReport`; sets `__all__` to exactly those seven names
(tested by API-1). `__version__` is read by hatchling from this file, so the version is written in
one place (TECH-STACK §8).

### 3.7 Error handling

- Invalid input → `ValueError` at the boundary, before any computation, with the UX-COPY message.
- Undecided rule → `NotImplementedError` naming the rule and the decision ID (constitution rule 2).
- Nothing is caught inside the package; there is no retry, logging or fallback (API-4).
- NumPy floating-point warnings cannot arise on validated inputs (std > 0 removes the only
  division); a test runs the suite with warnings turned into errors.

### 3.8 Model interface for M2 (design only; not built in M1)

```python
class UncertaintyModel(Protocol):  # models/_protocol.py
    def fit(self, X, y, *, seed: int) -> "UncertaintyModel": ...
    def predict_samples(self, X) -> tuple[np.ndarray, np.ndarray | None]:  # (S, n), (S, n) | None
        ...
```

- `X` has shape `(n, m)`, any m ≥ 1 (MODEL-1). Each model takes `m` from `fit`.
- The evaluation layer, never the model, calls `total_std` (MODEL-2). How `noise_vars is None`
  reaches σ is HANDOFF Q-3 (M2 ADR); the M1 `total_std` signature does not change.
- `train_and_compare(X, y, *, seed, split=(0.6, 0.2, 0.2), models=None, **hyper)` orchestrates:
  split (own streams per part, DATA-1/DATA-5) → `fit` each → `fit_scaling` on the calibration
  part → `evaluate` on the test part → rank the three total-uncertainty models (MODEL-5a, P-09)
  → `ModelComparison(models, table, best)`. It imports the core; the core never imports it.
- Adding a model = one class that satisfies the protocol; no change to combine, metrics, report
  or compare (MODEL-4).

### 3.9 reproduce/ pipeline (built Oct 3; designed here)

```
uv run python -m reproduce all
   ├── evidence     : sha256 of the 5 + 12 notebooks and the PDF  → reproduce/EVIDENCE.md      REPRO-5a
   ├── provenance   : printed outputs of every notebook → reproduce/provenance/*.json          REPRO-5c (P-08)
   ├── published    : faithful DONN port, 10 seeded runs × 3 test sets → output/published/     REPRO-3, REPRO-4
   ├── corrected    : the correction ladder, rungs 1–4, each one change → output/corrected/    ERRATA E-1
   └── tables       : golden comparison against reproduce/published_values.csv → output/tables/ REPRO-2
```

- `reproduce/published/`: `data.py` (the thesis generator, verbatim, with `np.random.seed(0)`),
  `donn.py` (the `DeepONet` class and the thesis training loop), `run.py` (10 runs per test set;
  `torch.manual_seed(run)` and a NumPy `SeedSequence` per run). `DEVIATIONS.md` lists the four
  expected deviations (REPRO-4).
- `reproduce/corrected/`: imports `uqcalibrate`; rung 2 replaces only the σ line with
  `total_std`; rung 3 only the band with `z_value(0.95)`; rung 4 only the test generator with the
  corrected law (needs P-05).
- Every output file carries the seed, the package version and the command line (REPRO-1).
- Nothing under `src/` imports `reproduce/` (tested, REPRO-3).

## 4. Scale and reliability

- **Load:** a call on n = 10⁶ points allocates about six float64 temporaries per level
  (≈ 50 MB per level, freed level by level); on n = 10⁷ this is still under 1 GB. Above that, the
  user chunks; a streaming API is a parking-lot item.
- **Determinism:** no randomness in the core; `statistics.NormalDist` is pure Python and bit-stable
  across platforms; NumPy reductions on float64 may differ in the last bits across BLAS builds,
  which is why known-answer tests use `rtol=1e-12`, not exact equality.
- **Failure modes:** invalid input (handled at the boundary) and undecided rules (explicit
  `NotImplementedError`). There is no I/O, network or state, so nothing else can fail.
- **CI (TECH-STACK §9):** the default suite on five Python versions × two operating systems;
  the `repro` suite runs on demand.

## 5. Trade-offs

| Decision | Chosen | Alternative | Why |
|---|---|---|---|
| Quantiles | `statistics.NormalDist` | SciPy | zero dependencies (API-5); accuracy to 1e-15 is more than the 1e-12 oracles need |
| Verdict interval | Wilson | Clopper–Pearson; normal approximation | Wilson is defined at k = 0 and k = n and has better average coverage than Clopper–Pearson; the normal approximation fails at the edges |
| Validation | once at the public boundary; private kernels trust inputs | validate in every function | avoids re-checking 10⁶ points four times per report; kernels are private so the trust never leaks (ADR-0003) |
| Report object | frozen dataclass with tuple fields | dict; mutable class | immutability is API-4; a dataclass gives `__eq__`, `__repr__` and typing for free |
| `fit_scaling` output | the factor | the scaled `std` array | the factor is a number the user can log, report and apply to new data |
| Coverage result | a `float` | per-point mask | the mask is one line for the user (`_bounds` is private); one obvious return type |
| Pending rules | `NotImplementedError` at the exact line | best-guess defaults | constitution rule 2; a wrong default would ship silently |

## 6. What to revisit as the package grows

- **Multi-output targets** (parking lot): `y` of shape `(n, d)` would add an axis to every
  kernel; the validation layer is the only place that would need a policy.
- **Input-dependent scaling** (MISSION §4): a `fit_scaling(..., groups=...)` variant, fitted per
  region, once someone needs it.
- **CRPS** (MATH-4 "where available"): one more kernel in `metrics.py` and one more column.
- **Report formats**: `to_markdown()` / `to_html()` are cheap to add on top of `to_dict()`;
  `str()` stays ASCII.
- **Streaming or chunked evaluation** for n beyond memory.
- **The models extra** (M2): the protocol above; GPU support is out of scope for v1.
