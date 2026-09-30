# Handoff: uqcalibrate M1 (NumPy core)

**Phase 1, step 3 · 2026-09-30.** From the notebooks and MISSION.md to the module layout in
TECH-STACK.md §3. Everything a developer needs to build M1 without guessing; where something is
not decided, it is listed under *Open questions*, and the code raises `NotImplementedError` there.

## Overview

M1 turns three notebook idioms into six pure functions: combining sampled predictions into one
standard deviation (`total_std`), counting points inside a band (`coverage`), and scoring a band
(`gaussian_nll`, `interval_score`), plus the two functions the notebooks never had: a verdict
(`evaluate`) and a fix (`fit_scaling`). The notebooks' training code, data generators and plots
do **not** migrate into the package: training moves to `uqcalibrate.models` in M2, and the
published data generator and the published σ formula move to `reproduce/` (REPRO-3).

## Layout

See TECH-STACK.md §3. In M1 the package is `src/uqcalibrate/` with `__init__.py`,
`_validate.py`, `_normal.py`, `combine.py`, `metrics.py`, `calibration.py`, `report.py`.
No file under `src/` imports PyTorch, `reproduce/` or `tests/`.

## Constants ("design tokens")

| Constant | Value | Where used | Rule |
|---|---|---|---|
| z(0.50) | 0.6744897501960817 | bands at 50% | MATH-2 |
| z(0.80) | 1.2815515655446008 | bands at 80% | MATH-2 |
| z(0.90) | 1.6448536269514715 | bands at 90% | MATH-2 |
| z(0.95) | 1.9599639845400536 | bands at 95%, and the Wilson interval's own confidence | MATH-2, MATH-7 |
| ½·log(2π) | 0.9189385332046727 | NLL constant | MATH-5 |
| default `levels` | (0.5, 0.8, 0.9, 0.95) | `evaluate` | MATH-4 |
| Wilson confidence | 95%, fixed | verdicts | MATH-7 |

All z-values come from `statistics.NormalDist().inv_cdf((1 + p) / 2)` at run time; the numbers
above are the test oracles (to 1e-12). Nothing is hard-coded except the default levels.

## Migration map: notebook cell → function → rule

Cell indices are 0-based among code cells. "Defect" means a behavior the package must *not*
reproduce; each one is listed in ERRATA.md.

| Notebook, cell | What the cell does | Becomes | Rule | Notes and defects |
|---|---|---|---|---|
| `Refactor_donn` 2 | `start=-7, end=7, n=300, batch_size=16, modelSamples=50` | `reproduce/published/` constants (M1, Oct 3); model hyperparameters in `models/` (M2) | REPRO-4 | Not packaged. |
| `Refactor_donn` 3 | `sample_dataset(start, end, n)`: `np.random.seed(0)`; x = 300 evenly spaced points; mean sin(x/2); noise std `abs((abs(start)+abs(end))/2 - abs(x))/16`; `stats.norm(...).rvs(random_state=0)` | `reproduce/published/data.py` (faithful) and, once P-05 is decided, the corrected generator (DATA-3) | DATA-1, DATA-3, DATA-5, REPRO-4 | **Defect:** the noise law depends on the sampling range (E-3). **Defect:** the variable `sample_var` holds standard deviations (API-3). The training law gives σ = 0 at x = ±7 exactly (two of the 300 training points). |
| `Refactor_donn` 4 | `x_test, y_test = sample_dataset(-10, 10, 200)` | same as above | DATA-1 | **Defect:** the test set's noise law differs from training and reuses `random_state=0`. |
| `Refactor_donn` 5, 6 | In-range mask `-7 <= x <= 7` (140 points); out-of-range mask (60 points) | `reproduce/`: the two subsets of the corrected benchmark | DATA-3 | Boundaries inclusive on the in-range side. Verified: 140 + 60 = 200. |
| `Refactor_donn` 7 | Tensors and `DataLoader`s; the test loaders shuffle | `models/` (M2) | MODEL-1 | Shuffling a test loader is harmless for averaged metrics. |
| `Refactor_donn` 8 | `DeepONet`: hidden 35; branch on x, trunk on fresh `torch.normal(0, 1)` noise each forward pass; `mu` linear; `var = exp(...)` | `models.DeepOperatorNeuralNetwork` (M2) and the faithful copy in `reproduce/published/` (M1, Oct 3) | MODEL-1, MODEL-6, REPRO-4 | No prior, posterior or KL term (E-6). The noise draw is what makes 50 passes differ. |
| `Refactor_donn` 9, lines 4–9 | 50 forward passes; `mus` and `vars` lists | the caller's job: `means, noise_vars` of shape (S, n) | MODEL-1 | `noise_vars` here are variances (`exp` output), correctly named for once. |
| `Refactor_donn` 9, line 11 | `means = stack(mus).mean(axis=0)` | `means.mean(axis=0)`, computed by the caller (README) | — | Correct as written. |
| `Refactor_donn` 9, lines 12–13 | `stds = stack(mus).std(axis=0)**(1/2) + stack(vars).mean(axis=0)**(1/2)` | **`combine.total_std(means, noise_vars)`** = `sqrt(noise_vars.mean(axis=0) + means.var(axis=0))` | **MATH-1** | **The thesis error (E-1):** square root of a standard deviation, and standard deviations added. Also `torch.std` divides by S−1; MATH-1a (P-01) divides by S. A test feeds the 3-4-5 inputs to both formulas. |
| `Refactor_donn` 9, lines 17–18 | `lower = mu - 2*sigma`, `upper = mu + 2*sigma` | `metrics._bounds(mean, std, level)` with z = Φ⁻¹((1+p)/2) | **MATH-2** | **Defect:** ±2σ is a 95.45% band (E-4). |
| `Refactor_donn` 9, lines 20–21 | `count_within = sum((y >= lower) & (y <= upper))`; percentage ×100 | **`metrics.coverage(y, mean, std, level)`**, a fraction in [0, 1] | **MATH-3** | Inclusive bounds, as in the notebook. Reports print percentages. |
| `Refactor_donn` 9, lines 22–46 | prints; seaborn band plot; returns the percentage | nothing in the core (API-4); plots in `reproduce/` only | API-4 | — |
| `Refactor_donn` 10, lines 3–7 | `GaussianNLLLoss(eps=1e-02)`, Adam, lr 1e-3, 150 epochs | `models/` training (M2); faithful copy in `reproduce/published/` | MATH-5a, REPRO-4 | Training keeps PyTorch's loss (no constant, clamped at eps). Scoring uses **`metrics.gaussian_nll`** (full NLL, no clamp). |
| `Refactor_donn` 10, lines 11–33 | Training loop; test loss and MSE computed every 10 epochs (last at epoch 140); `loss` and `mse_train` are the last mini-batch's | `reproduce/published/` | REPRO-4 | Confirms two "under verification" items for the DONN: the reported training loss is one mini-batch, and the reported test metrics are from epoch 140 while the plotted model is from epoch 149. Other notebooks: check on Oct 3. |
| `Refactor_donn` 12 | Ten runs per test set; pickles; pandas table; `to_csv` | `reproduce/published/run.py` (10 seeded runs) and REPRO-2b | REPRO-1, REPRO-2 | The cell does not parse as saved (REPRO-4, deviation 2). |
| `Refactor_snn` 9 | `sigma = var ** (1/2)` from a single network | `total_std(means[None, :], noise_vars[None, :])`, S = 1 | MATH-1 (S = 1 case) | Correct as written; the epistemic term is 0 at S = 1 under P-01. |
| `Refactor_denn` 10 | `stds = stack(mus).std(axis=0) ** (1/2)`; `vars` collected but never read | `total_std` over the 5 members | MATH-1 | **Defect:** noise term dropped and the extra square root (E-2). |
| `Refactor_bnn` 11, `Refactor_mcdnn` 10 | `stds = preds.std(axis=0)` over 500 / 50 samples | `total_std(means, noise_vars=None)`: epistemic only | MATH-1, MODEL-3, MODEL-6 | No noise term exists in these architectures. How `None` is handled is an M2 ADR question (Q-3 below). |
| `Refactor_mcdnn` 11 | `all_test_losses.append(loss.item())` | nothing | — | **Defect:** the training loss is recorded as the test loss (E-7). |

## Components (the six functions and the report)

| Function | Signature | Returns | Variants / states | Rule |
|---|---|---|---|---|
| `total_std` | `(means, noise_vars)`; both `(S, n)`, S ≥ 1 | `ndarray (n,)`, float64 | S = 1 gives `sqrt(noise_vars[0])` | MATH-1 |
| `coverage` | `(y, mean, std, level=0.95)`; `std` `(n,)` or scalar | `float` in [0, 1] | inclusive bounds | MATH-2, MATH-3 |
| `gaussian_nll` | `(y, mean, std)` | `float` | no clamp; σ > 0 enforced by API-2 | MATH-5 |
| `interval_score` | `(y, mean, std, level=0.95)` | `float` (mean over points) | α = 1 − level | MATH-2, MATH-6 |
| `evaluate` | `(y, mean, std, levels=(0.5, 0.8, 0.9, 0.95))` | `CalibrationReport` | raises `NotImplementedError` if 0.95 is not in `levels` (Q-1) | MATH-4, MATH-7 |
| `fit_scaling` | `(y, mean, std)` | `float` > 0 | validates, then raises `NotImplementedError` naming MATH-8 until P-04 is decided; all-zero residuals raise `ValueError` first | MATH-8 |
| `CalibrationReport` | frozen dataclass | — | `str()` per UX-COPY.md; `to_dict()` | §6 |
| `compare` *(added with P-09)* | `(y, predictions, epistemic_only=None, levels=...)` | `ModelComparison` | ranks by NLL; `best` is `None` when the lead is within 2 SE; needs n ≥ 2 | MODEL-5a, MODEL-3, MATH-4 |

Every function: converts inputs with `np.asarray(..., dtype=float)`, validates per API-2, and
draws no random numbers (API-4).

## States

**Verdicts (per level):** `OVERCONFIDENT` (Wilson upper < level), `UNDERCONFIDENT` (Wilson
lower > level), `CONSISTENT` (otherwise). Headline = verdict at 0.95.

**Errors:** `ValueError` with the messages in UX-COPY.md. `NotImplementedError` for the pending
paths, naming the rule and the decision ID. `TypeError` is never raised on purpose; a non-numeric
input fails in `np.asarray` and is reported as non-finite or non-convertible by the validator.

**Pending paths in M1 code (each raises `NotImplementedError`):**
- `fit_scaling` objective: MATH-8, P-04.
- `evaluate` headline when 0.95 is not among `levels`: MATH-7, Q-1.

## Edge cases

| Case | Behavior | Rule |
|---|---|---|
| n = 1 | Allowed everywhere. Coverage is 0 or 1; the Wilson interval is very wide; verdict CONSISTENT. | MATH-3, MATH-7 |
| `std` scalar | Broadcast to `(n,)`. | §6 |
| all points inside / all outside | coverage 1.0 / 0.0; Wilson works at k = 0 and k = n (that is why Wilson, not the normal approximation). | MATH-3, MATH-7 |
| point exactly on a bound | counts as inside. | MATH-3 |
| `level` 0 or 1 | `ValueError`. | MATH-2 |
| `levels` empty, duplicated, or unordered | empty: `ValueError`; duplicates: `ValueError`; order kept as given (D-2 below). | — |
| `std` contains 0, negative, NaN, inf | `ValueError` naming `std` (E-a). | API-2 |
| `noise_vars` negative | `ValueError`; zero is allowed. | API-2 |
| shapes disagree | `ValueError` naming both shapes (E-b); the `(S, n)`-for-`(n,)` case has its own message. | API-2 |
| few points | verdict CONSISTENT with a wide plausible range; the footer line explains (E-c). | MATH-7 |
| all residuals zero in `fit_scaling` | `ValueError` (E-d). | MATH-8 |
| `noise_vars is None` | not in M1; M2 ADR (Q-3). | MODEL-3 |
| integer or list inputs | converted to float64. | API-2 |
| very large n (10⁶) | vectorized NumPy; no Python loops over points. | — |

## Accessibility and output

`str(report)` is ASCII, under 80 columns, fixed-width. `to_dict()` returns `int`, `float`,
`str`, `list` and `tuple` only, so it serializes to JSON without a custom encoder.

## Acceptance tests (planned names; each carries `@pytest.mark.rule`)

| Rule | Test | Kind |
|---|---|---|
| MATH-1 | `test_math1_three_four_five`, `test_math1_two_member_unequal_noise`, `test_math1_single_member_returns_sqrt_noise`, `test_math1_rejects_thesis_formula` (asserts ≠ 6.0598 on the 3-4-5 inputs), `test_math1_divisor_is_population` (ddof=0 vs ddof=1 on 5 members) | known-answer |
| MATH-2 | `test_math2_z_values` (four oracles to 1e-12), `test_math2_never_two_sigma` (a band at 0.95 is narrower than ±2σ), `test_math2_level_out_of_range_raises` | known-answer, boundary |
| MATH-3 | `test_math3_all_inside_all_outside_boundary`, `test_math3_oracle_reaches_nominal_coverage` (n = 100,000, seeded, ±0.005 at each default level) | known-answer, statistical |
| MATH-4 | `test_math4_evaluate_reports_three_things_per_level`, `test_math4_inflated_sigma_wins_coverage_loses_width_and_nll` | property |
| MATH-5 | `test_math5_constant` (y = μ, σ = 1 gives 0.9189385332046727), `test_math5_minimum_at_true_sigma` (seeded sweep) | known-answer, property |
| MATH-6 | `test_math6_inside_scores_width`, `test_math6_known_miss`, `test_math6_minimum_at_honest_sigma` | known-answer, property |
| MATH-7 | `test_math7_wilson_known_answers` (183/200 and 186/200), `test_math7_oracle_consistent_at_large_n`, `test_math7_halved_sigma_overconfident`, `test_math7_small_n_consistent`, `test_math7_headline_needs_95_raises_not_implemented` | known-answer, statistical, boundary |
| MATH-8 | `test_math8_pending_raises_not_implemented`, `test_math8_zero_residuals_raise_value_error`; the acceptance test `test_math8_recovers_factor_two` is written now and marked `xfail(reason="P-04 pending")` | boundary, statistical |
| API-1 | `test_api1_public_names_exact` (`dir(uqcalibrate)` public names == the seven) | boundary |
| API-2 | one test per row of the UX-COPY error table, matching the message text | boundary |
| API-3 | `test_api3_parameter_names_carry_units` (signature inspection) | boundary |
| API-4 | `test_api4_pure_no_print_no_random` (captures stdout; same inputs, same outputs; global RNG state unchanged) | property |
| API-5 | `test_api5_imports_without_torch` (imports `uqcalibrate` with `torch` blocked in `sys.modules`), `test_api5_core_imports_only_numpy_and_stdlib` (AST scan of `src/`) | boundary |
| API-6 | `test_api6_no_abbreviations_in_core_strings` (no "DONN"/"SNN"/... in `src/` strings) | boundary |
| MODEL-5, MODEL-3 *(added with P-09)* | `test_model5a_orders_by_nll_after_scaling`, `test_model5a_never_ranks_by_coverage_alone`, `test_model5a_identical_models_are_too_close_to_call`, `test_model5a_paired_lead_known_answer` (log 2, SE 0), `test_model5a_epistemic_only_listed_separately_and_never_best`, table, `to_dict`, purity, messages | known-answer, property, boundary |
| G5 | `test_meta_every_in_force_rule_has_a_test` (parses MISSION.md rule IDs in force for M1, compares with markers) | meta |

DATA-1, DATA-2 and DATA-5 get their tests with the corrected generator on Oct 3 (they need
`reproduce/data.py`, which needs P-05); until then the meta-test's M1 list excludes them and
says why.

## Decisions taken in this handoff (for Carol's veto, not blocking)

- **D-1** `total_std` requires 2-D input; it does not silently promote a 1-D array to S = 1,
  because a 1-D `means` is exactly the mistake API-2's second example describes.
- **D-2** `levels` are reported in the order given; duplicates raise `ValueError` because two
  identical rows would print the same verdict twice and `to_dict()` would carry a redundant key.
- **D-3** The Wilson interval's confidence is fixed at 95% and not a parameter: MATH-7 names
  one interval, and a knob would invite tuning the verdict.
- **D-4** `fit_scaling` ran the full API-2 validation and the all-zero-residuals check before
  raising `NotImplementedError`; superseded when P-04 was decided (the formula is in).
- **D-5** *(P-09)* The ranking rule lives in the core as `compare`, not in the PyTorch extra,
  because it needs no PyTorch and serves users who bring several models' predictions;
  `train_and_compare` (M2) calls it, so the rule exists once.

## Open questions

- **Q-1 (MATH-7, needed before release).** Headline when `levels` does not include 0.95:
  raise (current), use the highest requested level, or require 0.95 always?
- **Q-2 (P-05 to P-08).** Pending register decisions for the erratum track (P-04 and P-09 were
  decided on 2026-09-30); P-05, P-07, P-08 gate the DONN reproduction, not the package.
- **Q-3 (MODEL-3, M2 ADR).** How σ is formed for a model whose `noise_vars` is `None`:
  `total_std(means, None)` treating the noise term as zero and labeling the model, or the
  evaluation layer passing zeros. The M1 signature is unchanged either way.
- **Q-4 (DATA-3, with P-05).** The training law gives σ = 0 at x = ±7 exactly. Keep it (faithful
  to the thesis) or floor it (say 1e-3) in the corrected benchmark?
