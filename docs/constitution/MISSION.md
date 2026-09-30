# MISSION — uqcalibrate

**Status:** DRAFT constitution v1 · 2026-09-30 · amended the same day (§10) · awaiting Carol's approval
**Authority:** This document is normative. When code and this document disagree, the code is
wrong. TECH-STACK.md governs *how* things are built; ROADMAP.md governs *when*.

**How to read this document**

- Every rule has an ID (MATH-n, DATA-n, REPRO-n, API-n, MODEL-n). Every change, test and review
  comment names the ID it serves.
- A rule or clause tagged **⏸ PENDING CAROL'S REVIEW** is not in force. Code that reaches it
  raises `NotImplementedError("<ID> is pending Carol's review")`. It never guesses.
- Amendments are appended to §10 with a date and a reason. Changing a MATH rule needs Carol's
  approval.

---

## 1. Problem

A regression model that reports uncertainty makes a promise: "95% of outcomes will land inside
this band." Few users check whether the promise is kept, and the arithmetic that builds the band
fails silently when it is wrong.

This package exists because that happened in a published thesis (Gudumotu, CSULB, 2024). The
total predictive standard deviation was computed as `√std(μ) + √mean(σ²)` instead of
`√(var(μ) + mean(σ²))`. That inflated the bands about 1.9× and produced a headline result (89.05%
coverage, Table 32) that the architecture did not earn. The scoreboard used, coverage alone, could
not catch it: wider bands always catch more points.

## 2. Users

- **U1 Practitioner.** Has a regression model, in any framework, that outputs a mean and a
  standard deviation per prediction. Wants to know whether the uncertainty can be trusted, and to
  fix it if not.
- **U2 Student or researcher comparing uncertainty methods.** Needs one correct way to combine
  uncertainties and one fair scoreboard.
- **U3 Reviewer of the thesis.** Wants to regenerate the published and the corrected numbers with
  one command.

## 3. Goals

- **G1** A U1 user with three arrays (y, mean, std) gets a calibration verdict in one call, with
  NumPy as the only dependency.
- **G2** A model whose bands are off by a constant factor is fixed on held-out data in one call,
  and re-evaluation shows the fix.
- **G3** Exactly one function computes total predictive variance. No public path computes it any
  other way.
- **G4** One command regenerates every corrected number in ERRATA.md, and regenerates the published
  DONN numbers (the headline) from a faithful copy of the published computation. Every other
  published number is cited with its table, page and provenance (REPRO-3, REPRO-5d).
- **G5** Every rule in force has at least one test that names it.
- **G6** (M2) A user with only X and y gets the five thesis models trained, each calibrated on a
  held-out part and scored on a separate test part, in one call, with a ranking that follows
  MATH-4 and MODEL-3 (the three total-uncertainty models are ranked; the two epistemic-only
  models are shown in their own section, MODEL-6), and the fitted winner ready to predict.

## 4. Non-goals (out of scope for v1)

- **Training models in M1.** The core evaluates predictions. Models arrive in M2 as an optional
  PyTorch extra.
- **Non-Gaussian predictive distributions** (quantile, mixture or sample outputs as the final
  prediction). v1 assumes a Gaussian predictive distribution per point.
- **Multi-output targets** (y of shape (n, d)). Targets are 1-D. Inputs X may have any number of
  features; the core never sees X (story 6).
- **Input-dependent calibration.** `fit_scaling` fits one factor for the whole input space. A
  factor that varies with x (for example in-range versus out-of-range) is in the ROADMAP parking
  lot.
- **Classification calibration** (classifier ECE, temperature scaling). A different problem.
- **CRPS, conformal intervals, plots.** Parking lot in ROADMAP.md.

## 5. User stories

1. As U1, I call `evaluate(y, mean, std)` and see promised versus delivered coverage at 50, 80, 90
   and 95%, the mean band width, the NLL and the interval score, with a verdict.
2. As U1 with an overconfident model, I call `fit_scaling` on a held-out calibration split,
   multiply my std by the returned factor, and `evaluate` on separate test data confirms the fix.
3. As U2 with an ensemble or a sampled model, I pass per-member means and noise variances to
   `total_std` and get the one correct standard deviation.
4. As U2 comparing models, I never see a model ranked by coverage alone.
5. As U3, I run one command and get the published DONN numbers regenerated, every table regenerated
   with the corrected computation, and a list of the published numbers that could not be traced to
   a surviving run.
6. As U1 with a regression on m input features (for example the three-input Ishigami function), I
   make the same calls as in stories 1 to 3. The core takes y, mean and std, one value per point,
   and never sees X, so the number of input features never changes a call. In M2 the built-in
   models accept X of shape (n, m) for any m ≥ 1 (MODEL-1), and the corrected Ishigami rerun is
   the README's multi-feature example.
7. As U1 without a model of my own (M2), I call `train_and_compare(X, y)` (name pending Phase 1
   step 2). The package splits my data into training, calibration and test parts, trains the five
   thesis models, fixes each one's bands on the calibration part (MATH-8), scores each on the test
   part, ranks the three models that report total uncertainty in one table with coverage, width
   and NLL (MATH-4), shows the two epistemic-only models in a separate section (MODEL-3, MODEL-6),
   tells me when two ranked models are too close to call, and hands me the fitted winner
   (MODEL-5).

**Edge cases**

- **E-a** std contains 0, a negative value, NaN or inf: an error names the argument and the likely
  cause (for example, variances passed where standard deviations belong).
- **E-b** Shapes disagree: an error names both shapes.
- **E-c** Few points: the verdict says the evidence is too weak, instead of claiming calibration.
- **E-d** All calibration residuals are zero: `fit_scaling` refuses, because no positive finite
  factor exists.
- **E-e** A model has no noise output: it is labeled "epistemic only" and never ranked against
  total-uncertainty models (M2).

## 6. API contract (M1 public surface)

`import uqcalibrate as uqc`. Exactly these names are public in M1 (API-1).

| Function | Signature | Returns | Rules |
|---|---|---|---|
| `total_std` | `total_std(means, noise_vars)` | ndarray, shape (n,) | MATH-1 |
| `coverage` | `coverage(y, mean, std, level=0.95)` | float in [0, 1] | MATH-2, MATH-3 |
| `gaussian_nll` | `gaussian_nll(y, mean, std)` | float | MATH-5 |
| `interval_score` | `interval_score(y, mean, std, level=0.95)` | float | MATH-2, MATH-6 |
| `evaluate` | `evaluate(y, mean, std, levels=(0.5, 0.8, 0.9, 0.95))` | `CalibrationReport` | MATH-4, MATH-7 |
| `fit_scaling` | `fit_scaling(y, mean, std)` | float > 0 | MATH-8 |
| `compare` | `compare(y, predictions, epistemic_only=None, levels=(0.5, 0.8, 0.9, 0.95))` | `ModelComparison` | MATH-4, MODEL-3, MODEL-5a |

**Shapes.** `means` and `noise_vars` have shape (S, n): S ≥ 1 samples or ensemble members on axis
0, n points on axis 1. For a single network, S = 1. `y` and `mean` have shape (n,). `std` has
shape (n,) or is a scalar.

**CalibrationReport.** Immutable. Fields: `n`, `levels`, `delivered`, `plausible_ranges`,
`verdicts`, `mean_widths`, `interval_scores`, `nll`, `headline`. `delivered`, `plausible_ranges`
(one `(low, high)` pair per level), `verdicts`, `mean_widths` and `interval_scores` hold one
entry per level, in the order of `levels`, and are named in the plural for that reason; `n`,
`nll` and `headline` are single values. Coverages are fractions in [0, 1] (MATH-3) and are
printed as percentages. `str(report)` renders the report below, ASCII only; `report.to_dict()`
returns plain Python types.

Report wording (set in Phase 1 step 2, docs/design/UX-COPY.md). The numbers are a real run:
200 seeded points whose model reports half the true spread (README):

```
uqcalibrate: 200 points, Gaussian bands
promised  delivered  plausible range  verdict        mean width  interval score
     50%      27.0%    21.3% - 33.5%  OVERCONFIDENT        0.81            3.34
     80%      45.5%    38.7% - 52.4%  OVERCONFIDENT        1.54            5.50
     90%      55.5%    48.6% - 62.2%  OVERCONFIDENT        1.98            7.73
     95%      65.0%    58.2% - 71.3%  OVERCONFIDENT        2.35           10.88
NLL 2.39
plausible range: 95% Wilson interval for the true coverage.
A wider range means weaker evidence.
OVERCONFIDENT: 95% promised, 65.0% delivered.
```

The headline for the other verdicts: `UNDERCONFIDENT: 95% promised, 99.5% delivered.` and
`CONSISTENT: 95% promised, 94.0% delivered. No evidence of miscalibration.`

**Target workflow**

```python
report = uqc.evaluate(y_test, mean_test, std_test)  # verdict: OVERCONFIDENT
s = uqc.fit_scaling(y_cal, mean_cal, std_cal)  # held-out calibration split
report = uqc.evaluate(y_test, mean_test, s * std_test)  # confirms the fix
```

**`compare` and `ModelComparison`** *(added by amendment 9, P-09)*. `predictions` maps a model
name to its `(mean, std)` on the test data, after any factor from `fit_scaling` has been applied;
these models report total uncertainty and are ranked. `epistemic_only` maps names to `(mean, std)`
for models whose bands hold no noise term; they are shown in their own section and never ranked
(MODEL-3). `ModelComparison` is immutable, with fields `n`, `names` (ranked, best first),
`reports` (one `CalibrationReport` per ranked model), `best` (a name, or `None` when the top two
are too close to call), `runner_up`, `lead`, `lead_se` (the leader's mean per-point NLL advantage
over the runner-up and its standard error), `epistemic_only_names`, `epistemic_only_reports`,
`headline`; `str()` renders the table below (ASCII, at most 100 columns with the thesis names);
`to_dict()` returns plain Python types.

Output of `compare` (a real run: 60 seeded points, three models at 0.5, 0.9 and 1.0 times the
true spread, one epistemic-only model at 0.35):

```
uqcalibrate: 3 models ranked, 1 epistemic-only, 60 points
rank  model                               95% delivered  mean width      NLL  verdict
   1  deep operator neural network                95.0%        4.52     1.55  CONSISTENT
   2  deep ensemble neural network                95.0%        5.03     1.55  CONSISTENT
   3  simple neural network                       73.3%        2.51     2.19  OVERCONFIDENT
Too close to call: the top two differ by 0.00 +/- 0.02 NLL per point.

Epistemic only (bands hold no noise term; not ranked)
   -  Monte Carlo dropout neural network          48.3%        1.76     3.69  OVERCONFIDENT
```

The other headline reads `Best: <name> (leads the runner-up by 0.06 +/- 0.02 NLL per point).`;
with one ranked model, `Best: <name> (the only model ranked).`

**Planned for M2 (not public in M1):** `uqcalibrate.models.train_and_compare(X, y, ...)`, which
trains the five thesis models and calls `compare` (MODEL-5), printing the same table with its own
first line about the split. It lives in the PyTorch extra. The M1 surface does not change.

---

## 7. Rules

### MATH: how uncertainty is combined, bounded and scored

**MATH-1 Total predictive variance.** *(seed rule)*
Given S ≥ 1 predictive samples (Monte Carlo passes, posterior samples or ensemble members) at each
of n points, with means μ[s, i] and noise variances v[s, i] ≥ 0:

```
μ̄[i]   = (1/S) · Σ_s μ[s, i]
Var[i] = (1/S) · Σ_s v[s, i]  +  (1/S) · Σ_s (μ[s, i] − μ̄[i])²
            aleatoric term          epistemic term
σ[i]   = √Var[i]
```

Variances are added and one square root is taken last. Standard deviations are never added.
Exactly one function computes this (`total_std`), and every model and every caller in the package
uses it (MODEL-2).

- **Clause 1a (divisor of the epistemic term).** *(decided by Carol, 2026-09-30: P-01)* The
  epistemic term uses 1/S (population variance, NumPy `ddof=0`). Reasons: it is the exact
  variance of the equally weighted mixture of the S members (the law of total variance applied to
  that mixture, as in Lakshminarayanan et al., 2017), and S = 1 then gives an epistemic term of
  exactly 0. The alternative 1/(S−1) (`ddof=1`, PyTorch's default and the thesis code's) inflates
  the epistemic variance by S/(S−1): ×1.25 for a 5-member ensemble, ×1.02 for 50 passes, and it
  is undefined at S = 1.
- **Acceptance.** (1) The 3-4-5 case: means [μ−3, μ+3], noise variances [16, 16] give σ = 5
  exactly. (2) A two-member known answer with unequal noise variances. (3) S = 1 returns √v.
  (4) On the 3-4-5 inputs the thesis formula gives 6.0598; a test asserts `total_std` does not.

**MATH-2 Central intervals.** *(seed rule)*
The central interval at level p, with 0 < p < 1, is `[μ − z·σ, μ + z·σ]` with
`z = Φ⁻¹((1 + p)/2)`, where Φ⁻¹ is the standard normal quantile function. "95%" means
z = Φ⁻¹(0.975) ≈ 1.959964. The package never uses ±2σ (that is a 95.45% band). ±2σ appears only
inside reproduce/, as part of the published computation (REPRO-3).
- **Acceptance.** z(0.95) = 1.9599639845400536, z(0.90) = 1.6448536269514715,
  z(0.80) = 1.2815515655446008, z(0.50) = 0.6744897501960817, each to 1e-12. A level outside
  (0, 1) raises.

**MATH-3 Coverage.**
`coverage = (1/n) · Σ_i 1{ lower[i] ≤ y[i] ≤ upper[i] }`, with inclusive boundaries, returned as a
fraction in [0, 1]. Reports display it as a percentage with one decimal.
- **Acceptance.** All points inside gives 1.0; all outside gives 0.0; a point exactly on a
  boundary counts as inside. Oracle test: y drawn from N(μ, σ²) with n = 100,000 (seeded) gives
  coverage within ±0.005 of p at each of the four default levels.

**MATH-4 Complete comparisons.** *(seed rule)*
Any output that summarizes or compares predictive uncertainty reports three things together:
coverage; mean band width (sharpness, the mean of 2·z·σ[i]); and at least one proper score, which
is always the Gaussian NLL, plus CRPS once it exists (ROADMAP parking lot). No public function
ranks or orders models by coverage alone.
- **Acceptance.** `evaluate` returns all three at every level. A model with σ = 1000 × the true σ
  reports near-100% coverage *and* a much larger width *and* a worse NLL than the honest model.

**MATH-5 Gaussian negative log-likelihood.**

```
NLL = (1/n) · Σ_i [ ½·log(2π·σ[i]²) + (y[i] − μ[i])² / (2·σ[i]²) ]
```

Natural log, mean over points. σ is never clamped, because clamping changes the score. σ[i] must
be > 0 (API-2).
- **Clause 5a (the constant).** *(decided by Carol, 2026-09-30: P-02)* The NLL includes
  ½·log 2π ("full" NLL), so values are comparable with the published uncertainty-quantification
  literature. PyTorch's `GaussianNLLLoss` omits the constant by default and clamps the variance at
  `eps`. The thesis losses used that form with `eps=1e-2`; it stays in reproduce/ and in model
  training (M2).
- **Acceptance.** y = μ with σ = 1 gives 0.9189385332046727. Over a sweep of scale factors, the
  NLL is lowest at the true σ (seeded).

**MATH-6 Interval score** (Gneiting & Raftery, 2007, §6.2).
For level p, let α = 1 − p and take [lower, upper] from MATH-2:

```
IS[i] = (upper[i] − lower[i])
      + (2/α) · (lower[i] − y[i]) · 1{y[i] < lower[i]}
      + (2/α) · (y[i] − upper[i]) · 1{y[i] > upper[i]}
```

Report the mean. Lower is better. The score charges for width and for misses, so neither inflating
nor shrinking the band improves it.
- **Acceptance.** A point inside scores exactly the width. A known-answer miss. For y drawn from
  N(0, 1), the mean score over a sweep of claimed σ is lowest at the honest σ (seeded).

**MATH-7 Calibration verdict.** *(decided by Carol, 2026-09-30: P-03)*
At each level p, compute the 95% Wilson score interval [L, U] for the true coverage, from k
points inside out of n. The verdict at p is OVERCONFIDENT if U < p (delivered significantly below
promised), UNDERCONFIDENT if L > p, and otherwise CONSISTENT (no evidence of miscalibration at
this sample size, which is not proof of calibration). The headline is the verdict at 95%, for
example "OVERCONFIDENT: 95% promised, 73.8% delivered". Every report shows the interval [L, U]
beside the delivered coverage as the "plausible range", with one fixed footer line defining it
("95% Wilson interval for the true coverage. / A wider range means weaker evidence.", two
lines), so a reader can see
how strong the evidence is without the package inventing a cut-off (edge case E-c). Wording:
docs/design/UX-COPY.md.
- **Why a statistical rule.** With n = 200, a perfectly honest 95% band delivers 95% ± 3 points
  by chance alone (one SD = 1.5 points). A fixed tolerance would call honest models miscalibrated
  at small n and miss real miscalibration at large n.
- **Rejected alternatives.** (b) A fixed tolerance, for example ±2 points. (c) The headline is
  the worst verdict across all four levels.
- **Headline when 95% is not among the requested levels:** undecided. `evaluate` raises
  `NotImplementedError` naming MATH-7 in that case (HANDOFF open question).
- **Acceptance.** An oracle at n = 100,000 is CONSISTENT at all levels. An oracle whose σ is
  halved is OVERCONFIDENT at all levels. With n = 10 the verdict is CONSISTENT unless the miss is
  extreme. Wilson known answer: k = 183 of n = 200 gives [0.868, 0.946] to three decimals, so the
  verdict at 95% is OVERCONFIDENT; k = 186 gives [0.886, 0.958], CONSISTENT.

**MATH-8 Variance scaling.** *(objective decided by Carol, 2026-09-30: P-04)*
`fit_scaling(y, mean, std)` returns one factor s > 0, fitted on held-out calibration data and
applied as `std_new = s · std`:

```
s = √( (1/n) · Σ_i ((y[i] − μ[i]) / σ[i])² )
```

This is the closed-form minimizer of the Gaussian NLL over s: no optimizer, no randomness. It is
the dataset-level form of the NLL's "confession" property: the fitted spread equals the observed
spread of standardized errors, so after scaling the standardized residuals have a root-mean-square
of exactly 1 on the calibration data. *Rejected alternative:* match coverage exactly at one chosen
level, s = quantile_p(|z|) / z_p, which is exact at that level and ignores the others.
- s is fitted on data disjoint from the data it is evaluated on. The function cannot detect
  reuse, so the docstring and every README example show a split. If all residuals are zero, it
  raises ValueError.
- **Acceptance.** Known answers: residuals exactly twice the std give s = 2; residuals 3 and 4
  with σ = 1 give √12.5. After scaling, the standardized residuals have RMS 1 (to 1e-12), and no
  other factor gives a lower NLL on the calibration data. For an oracle whose std is multiplied
  by 0.5, `fit_scaling` returns ≈ 2 on a large seeded calibration split, and `evaluate` on a
  separate test split is then CONSISTENT at all four levels.

### DATA: how benchmark data is generated

**DATA-1 One noise law, independent streams.** *(seed rule)*
Test data uses the same noise law as training data: the noise standard deviation is a fixed
function of the input, never recomputed from a sampling range or a split. Training, test and
calibration draws each use their own random stream, derived from one master seed
(`numpy.random.SeedSequence.spawn`).
*Thesis violation:* `sample_dataset` derives the noise from its `start`/`end` arguments, so the
(−10, 10) test set has larger noise than training at every shared x, and both sets use
`random_state=0`.

**DATA-2 Sampled inputs are never sorted.** *(seed rule)*
Inputs drawn independently are never sorted or reordered column by column. When a plot needs
ordered points, sort the row order once and apply that one permutation to every column.
*Thesis violation (as printed):* Figure 17 sorts x1, x2 and x3 separately, which makes them
perfectly rank-correlated.
- **Acceptance.** At n = 1,000 (seeded), |Spearman ρ| < 0.1 between any two generated input
  columns.

**DATA-3 Corrected 1-D benchmark.** **⏸ PENDING CAROL'S REVIEW**
Proposal: x_train is 300 evenly spaced points on [−7, 7]; x_test is 200 evenly spaced points on
[−10, 10], split into in-range |x| ≤ 7 (140 points) and out-of-range (60 points);
y = sin(x/2) + ε with ε ~ N(0, σ(x)²) and **σ(x) = |7 − |x|| / 16 for every x**. This is the
thesis's training law with its constant fixed at the training half-range and extended unchanged
beyond it (σ = 0.1875 at |x| = 10). Alternative: a law chosen to test out-of-range behavior on
purpose, which departs further from the thesis.

**DATA-4 Corrected Ishigami benchmark (M2).** **⏸ PENDING CAROL'S REVIEW**
Open points: (i) which design to correct: the thesis as printed (Figure 17: training range ±2π/3,
noise σ = 0.2·|f(x)|, sorted inputs) or the surviving notebooks (training range ±π/2, noise
σ = 0.2, unsorted inputs, hidden size 35); (ii) what "in range" means in 3-D.
*Provenance:* the notebook that produced Table 34 is lost. No surviving Ishigami notebook matches
the printed design or Table 34, and Carol confirmed on 2026-09-30 that her Google Drive holds no
other copies. Either design is therefore rebuilt from a description, not from the code that ran.
Proposal: correct the printed design, because it produced Table 34: independent inputs,
σ = 0.2·|f(x)|, training range ±2π/3, test range ±π, and "in range" when all three coordinates lie
inside the training box.

**DATA-5 Explicit randomness.**
Every random draw in tests and in reproduce/ comes from an explicitly seeded generator
(`numpy.random.default_rng`, `torch.Generator`). New code never relies on global random state.
The faithful copy of the published DONN computation keeps its original `np.random.seed(0)` calls,
plus added PyTorch seeding (REPRO-4). The M1 core draws no random numbers at all.

### REPRO: how published and corrected numbers are regenerated

**REPRO-1 One command.** *(seed rule)*
One command (planned: `uv run python -m reproduce all`) regenerates every table and figure used in
ERRATA.md, on CPU, with fixed seeds, into reproduce/output/: the corrected computation for every
table, and the published computation for the DONN (REPRO-3). Each output records its seed, the
package version and the command that made it. The runtime is documented.

**REPRO-2 Agreement with published numbers.** **⏸ PENDING CAROL'S REVIEW**
The seed rule reads: "the published computation must reproduce the published numbers within 0.05
points." Evidence gathered on 2026-09-30 shows this cannot be met for numbers that require
retraining:

- No notebook seeds PyTorch (`torch.manual_seed` appears nowhere). The weight initialization,
  batch order, dropout masks and DONN noise draws of the published runs cannot be replayed.
- The published 10-run means carry run-to-run noise. DONN full test: SD 1.60 (Table 2), so two
  honest 10-run reproductions differ by SD ≈ 0.72, and a ±0.05 window catches about 6% of them
  (about 3% for DONN out-of-range and for SNN full test).
- The saved notebook outputs match the published tables only for SNN and DENN. BNN, MCDO and DONN
  hold later re-runs (see ERRATA.md, Provenance).
- *Correction of earlier work:* THESIS_CRITIQUE.md reported the DONN full-test mean "reproduced to
  within 0.05 points" (89.10 vs 89.05). That match was within chance, not evidence of exactness:
  the same run was 0.67 points off out-of-range (97.67 vs 97.00) and about 3 points off for SNN
  full test (60.50 vs 63.45).

Proposal: split the rule in two.
- **REPRO-2a (deterministic, ±0.05).** Every summary number re-derived from the published
  per-run values (Tables 1–5, 11–15 and 21–25 from Tables 6–10, 16–20 and 26–30; Tables 31–33
  from their sources) matches within 0.05. Where notebook outputs survive (SNN, DENN), they match
  the published per-run values at printed precision.
- **REPRO-2b (retrained, statistical).** With 10 seeded runs per DONN table cell, each published
  mean lies within 3 standard errors of the difference, `3 · √(SD_pub²/10 + SD_rep²/10)`, of the
  reproduced mean (about ±2.1 points for DONN full test). Every gap is reported, pass or fail.
- **Scope (Carol, 2026-09-30).** REPRO-2b applies to the DONN only, the one published computation
  kept (REPRO-3). REPRO-2a needs no retraining and applies to every published table.

**REPRO-3 The published computation is quarantined, and kept for the DONN only.**
The published DONN computation (its σ formula, ±2σ bands, range-dependent test noise and
per-test-set retraining) exists only under reproduce/published/. The other published computations
(the SNN, DENN, BNN and MC dropout σ definitions; the sorted Ishigami inputs) are documented in
ERRATA.md and not re-implemented; their tables are regenerated with the corrected computation only
(Carol, 2026-09-30; ADR-0001). src/uqcalibrate/ never contains, imports or exposes a published
computation. reproduce/ may import uqcalibrate for the corrected computations. The package offers
only the corrected computation.
- **Acceptance.** A test asserts that no module under src/ imports reproduce. The MATH-1 test
  asserts the thesis formula's result is never returned.

**REPRO-4 Faithful copy, documented deviations.**
reproduce/published/ ports `Refactor_donn.ipynb` faithfully: same data generator, architecture,
hyperparameters, loops, σ formula, bands, loss and training details, including known defects.
Every deviation is listed in reproduce/DEVIATIONS.md with a reason. Expected: (1) explicit NumPy
and PyTorch seeding; (2) the last code cell rejoined where an assignment breaks after `=`: as
saved, that cell does not parse as Python, although it carries the outputs of two runs (verified
2026-09-30); (3) device-safe creation of the DONN noise tensor (no effect on CPU); (4) plots saved
to files instead of shown.

**REPRO-5 Evidence integrity and provenance.**
(a) reproduce/EVIDENCE.md records the SHA-256 of every evidence file: the 5 DNN notebooks, the
12 Ishigami notebooks and the thesis PDF. The originals are never modified.
(b) legacy/notebooks/ holds copies with outputs stripped. A test verifies that each copy's code
cells are byte-identical to its original's.
(c) **⏸ PENDING CAROL'S REVIEW:** before stripping, the printed text outputs of every notebook are
extracted to reproduce/provenance/ as small JSON files. They are the only evidence that the SNN
and DENN tables came from these notebooks, and that the BNN, MCDO and DONN tables did not.
(d) Every published number used as a golden value is transcribed from the PDF into
reproduce/published_values.csv with its table and page. Numbers with no surviving source run are
flagged "PDF only".

### API: how the public surface behaves

**API-1 Fixed public surface.** The M1 public names are the seven functions in §6 plus
`CalibrationReport` and `ModelComparison` (nine names). Everything else is private (leading
underscore). After this document is
approved, changing a public name, signature, default, return type or error type needs Carol's
approval.

**API-2 Validation.** Array-likes are converted to float64 NumPy arrays. Inputs must be finite:
NaN and inf raise; they are never silently dropped. std > 0 and noise_vars ≥ 0. Shapes follow §6.
Levels lie in (0, 1) and n ≥ 1. A violation raises ValueError with a message in three sentences:
what is wrong, with the number that proves it; what is required; the likely cause and the fix as
code (docs/design/UX-COPY.md holds the full set). Two examples:
*"std has 3 values <= 0 (min -0.2). Standard deviations must be positive. If these are variances
or log-variances, convert them first, e.g. std = np.sqrt(var)."*
*"mean has shape (50, 200) but y has shape (200,); mean must have shape (n,). If these are S
samples per point, combine them first: mean = means.mean(axis=0);
std = uqcalibrate.total_std(means, noise_vars)."*

**API-3 Units in names.** A parameter or variable is named for what it holds: `std` for standard
deviations, `var` or `noise_vars` for variances. *Thesis violation:* `sample_var` held standard
deviations.

**API-4 Pure functions.** Public functions do not print, write files, keep global state or draw
random numbers. The same inputs give the same outputs. `CalibrationReport` is immutable.

**API-5 Core dependencies.** The core uses NumPy and the Python standard library only. Normal
quantiles come from `statistics.NormalDist`. PyTorch appears only under `uqcalibrate.models` (M2,
optional extra). TECH-STACK.md §3 enforces this.

**API-6 Full model names in output.** *(Carol, 2026-09-30)* Every user-facing output (reports,
tables, error messages, docstrings) names a model by the thesis's full name, never an
abbreviation:

| Full name (display) | Python class (M2) | Thesis abbreviation, used only in ERRATA.md and reproduce/ |
|---|---|---|
| simple neural network | `SimpleNeuralNetwork` | SNN |
| deep ensemble neural network | `DeepEnsembleNeuralNetwork` | DENN |
| Monte Carlo dropout neural network | `MonteCarloDropoutNeuralNetwork` | MCDNN, MCDO |
| Bayesian neural network | `BayesianNeuralNetwork` | BNN |
| deep operator neural network | `DeepOperatorNeuralNetwork` | DONN |

One name per thing: a reader who sees a name in a table finds the same words in the code.

### MODEL: the contract every model follows (M2)

**MODEL-1 Output contract.** Every model implements `predict_samples(X)`, where X has shape (n, m)
for any number of input features m ≥ 1, and returns `(means, noise_vars)` with shape (S, n).
`noise_vars` is `None` when the model has no noise output. A model takes its number of input
features from the data or its constructor; no model assumes m = 1. Nothing else is required of a
model to be evaluated or compared.

**MODEL-2 One path to σ.** Models never combine uncertainty themselves. The evaluation layer calls
`total_std` on their outputs. *(seed rule: "every model uses it")*

**MODEL-3 Epistemic-only models.** *(seed rule)* A model whose `noise_vars` is `None` is labeled
"epistemic only". It is reported in its own section and never ranked against total-uncertainty
models.

**MODEL-4 Open for extension.** Adding a model requires no change to the code in combine, metrics,
report or compare.

**MODEL-5 Train, calibrate and compare in one call (M2).** `train_and_compare(X, y, ...)`:
(i) splits the data into training, calibration and test parts with a seeded shuffle (default
60/20/20; the caller may pass its own parts), each part with its own random stream (DATA-1,
DATA-5); (ii) trains each model on the training part only; (iii) fits each model's scaling factor
on the calibration part (MATH-8) and applies it; (iv) evaluates every model on the test part with
`evaluate` (MATH-4, MATH-7) through the core function `compare`; (v) returns the
`ModelComparison` from `compare`, plus the fitted models. The function orchestrates: it computes
no metric itself, and every σ comes from `total_std` (MODEL-2). Defaults are the thesis
hyperparameters; every one can be overridden.
- **Clause 5a (ranking rule).** *(decided by Carol, 2026-09-30: P-09; implemented once, in the
  core function `compare`, which `train_and_compare` calls)* Models that report total uncertainty
  are ordered by NLL on the test part after scaling, with coverage and mean width at 95% shown
  beside it and the full `evaluate` report kept for every model (MATH-4). The leader is named
  "best" only when its per-point NLL beats the runner-up's by more than two standard errors of
  the paired difference (mean and standard error of the per-point differences over the n test
  points); otherwise the headline says the top two are too close to call. Epistemic-only models
  are listed in a separate section and are never best (MODEL-3). `compare` needs n ≥ 2 points, so
  the standard error exists.
- **Acceptance.** A model made deliberately overconfident never ranks first. A model with
  σ × 1000 never ranks first despite 100% coverage. Two copies of the same model give "too close
  to call" with a lead of exactly 0. Known answer: two models at the truth with σ = 1 and σ = 2
  differ by exactly log 2 per point with zero standard error, so the σ = 1 model is best. An
  epistemic-only model with the best NLL of all is still not best and not in `names`.

**MODEL-6 Thesis architectures, unchanged (M2).** *(decided by Carol, 2026-09-30; ADR-0002)*
The five built-in models keep the thesis architectures. The Bayesian network and MC dropout
output a mean only, as in the thesis, so their bands hold no noise term (ERRATA E-2) and MODEL-3
applies: `train_and_compare` trains all five, ranks the three that report total uncertainty (the
simple, deep ensemble and deep operator neural networks), and lists the other two in a
separate epistemic-only section with the same columns and a note that their bands hold no noise
term. `.best` is chosen among the ranked three. No noise-head option exists in v1; adding one is
a ROADMAP parking-lot item. How σ is formed for a model whose `noise_vars` is `None` is settled in
the M2 architecture ADR; the M1 signature of `total_std` does not change.
- **Acceptance.** On a seeded problem, `train_and_compare` returns exactly three ranked rows and
  two epistemic-only rows, and `.best` is never an epistemic-only model.

---

## 8. Acceptance criteria for M1

- [ ] Every MATH and API rule in force, plus DATA-1, DATA-2 and DATA-5, has at least one passing
      test that names it.
- [ ] The MATH-1 known answers pass (3-4-5, two-member, S = 1), and a test fails on the thesis
      formula.
- [ ] The oracle test (MATH-3) reaches nominal coverage at 50, 80, 90 and 95%.
- [ ] `evaluate` → `fit_scaling` → `evaluate` runs end to end on a seeded example (MATH-8).
- [ ] `compare` ranks three models correctly and lists an epistemic-only model separately
      (MODEL-5a, MODEL-3).
- [ ] REPRO-1: one command regenerates the M1 tables (DONN published and corrected).
- [ ] REPRO-2 (as decided) passes or reports every gap; REPRO-3 and REPRO-5 tests pass.
- [ ] Importing `uqcalibrate` succeeds with PyTorch unavailable (API-5).
- [ ] ERRATA.md E-1 contains regenerated numbers and Carol has approved them.

## 9. PENDING register: decisions for Carol

| ID | Decision | Recommendation |
|---|---|---|
| P-01 | MATH-1a: divisor of the epistemic term | **Decided 2026-09-30:** 1/S (`ddof=0`) |
| P-02 | MATH-5a: include ½·log 2π in the NLL | **Decided 2026-09-30:** yes (full NLL) |
| P-03 | MATH-7: verdict rule | **Decided 2026-09-30:** Wilson interval per level; headline at 95% |
| P-04 | MATH-8: scaling objective | **Decided 2026-09-30:** closed-form NLL minimizer |
| P-05 | DATA-3: corrected 1-D noise law | σ(x) = \|7 − \|x\|\| / 16 for every x |
| P-06 | DATA-4: corrected Ishigami design (M2; the Table 34 notebook is lost) | Printed design, minus the sorting |
| P-07 | REPRO-2: agreement tolerance | Split: 2a exact ±0.05, every published table; 2b within 3 standard errors, DONN only |
| P-08 | REPRO-5c: extract notebook outputs before stripping | Yes |
| P-09 | MODEL-5a: ranking rule for `train_and_compare` (M2) | **Decided 2026-09-30:** NLL after scaling, MATH-4 columns beside it, paired test for "too close to call"; implemented in the core `compare` |
| P-10 | MODEL-6: noise heads so all five models are comparable (M2) | **Decided 2026-09-30:** no. Thesis architectures kept; three models ranked, two shown as epistemic-only (ADR-0002) |

## 10. Amendment log

- **2026-09-30** v1 drafted from Carol's seed rules and the evidence verification of the same
  date. Not yet approved.
- **2026-09-30, amendment 1.** Carol's decisions of the same day. No MATH rule changed.
  (1) *Scope:* the published computation is kept for the DONN only; every other table is
  regenerated with the corrected computation only. Changed: G4, story 5, DATA-5, REPRO-1, the
  REPRO-2 proposal, REPRO-3, REPRO-4, §9 (P-07). Reason: the DONN σ is the error behind the
  headline (ERRATA.md E-1), so it is the one published computation the erratum must re-run; the
  other published σ definitions are stated exactly from the code (ERRATA.md E-2), and re-running
  them would not change a corrected number. Recorded as ADR-0001.
  (2) *Provenance:* Carol confirmed that her Google Drive holds no other copies of the notebooks,
  so the notebook behind Table 34 is lost. Changed: DATA-4, §9 (P-06).
  (3) *New evidence:* the last code cell of Refactor_donn.ipynb does not parse as saved. Changed:
  REPRO-4, expected deviation (2). The BNN `print(x_)` deviation left with the BNN port.
- **2026-09-30, amendment 2.** Carol's requirement that users can calibrate regressions with any
  number of input features, as in the Ishigami experiment. The draft already allowed this (§4,
  "the core never sees X") but did not say so where a user would look. Added story 6; MODEL-1 now
  states the shape of X and that no model assumes one feature; §4 names input-dependent
  calibration as out of scope for v1 (ROADMAP parking lot). No MATH rule changed.
- **2026-09-30, amendment 3.** Carol's proposal that users may also bring only X and y and have
  the package train the five thesis models, calibrate them and say which is best. Added G6, story
  7, the M2 line in §6, MODEL-5 and MODEL-6, and P-09 and P-10 in §9. The M1 surface and every
  MATH rule are unchanged. Reason: the M1 core already serves users who bring predictions; this
  adds the second way in, as one orchestrating function in the M2 extra.
- **2026-09-30, amendment 4.** P-10 decided by Carol: no noise heads. MODEL-6 rewritten as an
  in-force rule; G6 and story 7 say that three models are ranked and two are shown as
  epistemic-only. Reason (Carol's choice, recorded in ADR-0002): the package ships the thesis
  architectures as they were, so nothing in the one-call path is an architecture the thesis
  never tested.
- **2026-09-30, amendment 5.** Carol asked that output say "deep operator neural network", not
  "DONN". Added API-6 (full thesis names in every user-facing output, matching Python class
  names) and the illustrative `ModelComparison` in §6. MODEL-6 wording aligned. No MATH rule
  changed.
- **2026-09-30, amendment 6.** Carol approved P-01, P-02 and P-03 as recommended. MATH-1 clause
  1a, MATH-5 clause 5a and MATH-7 are now in force. MATH-7 gained two clauses that follow from the
  decision: the interval is shown beside the delivered coverage (E-c), and the headline level is
  undecided when 95% is not requested, so `evaluate` raises `NotImplementedError` there instead
  of guessing. Wilson known answers added to the MATH-7 acceptance list.
- **2026-09-30, amendment 7 (Phase 1 step 2).** From docs/design/CRITIQUE.md and UX-COPY.md:
  `CalibrationReport` gains `plausible_ranges` and the field semantics are stated; the report
  wording, its footer line and the three headlines are fixed and ASCII-only; API-2 gains the
  error-message template and a second example (samples passed as a mean); MATH-7 names the
  "plausible range" column. Function names unchanged. No MATH rule changed.
- **2026-09-30, amendment 8 (Phase 2 step 2, architecture critique).** `CalibrationReport`
  fields `mean_width` and `interval_score` renamed `mean_widths` and `interval_scores`: they
  hold one value per level, like their plural neighbours. The function `interval_score` keeps
  its name (it returns one number). Awaiting Carol's veto; no MATH rule changed.
- **2026-09-30, amendment 9.** Carol decided P-04 and P-09 as recommended ("let us code P-04 and
  P-09 now"). MATH-8's objective is in force (the closed-form NLL minimizer) with known answers
  added to its acceptance list. MODEL-5a is in force and, because the ranking rule needs no
  PyTorch, it is implemented once in the core as the public function `compare` returning
  `ModelComparison`; `train_and_compare` (M2) will call it. §6 gains the `compare` row, the
  `ModelComparison` fields and a real example; API-1 now names nine public names; §8 gains a
  `compare` item. MATH rules unchanged except MATH-8's decided clause.
