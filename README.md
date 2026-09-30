# uqcalibrate

Check whether a regression model's uncertainty is honest, and fix it when it is not.

A model that reports uncertainty makes a promise: "95% of true values will fall inside this
band." `uqcalibrate` checks that promise on your test data, scores it with numbers that cannot be
gamed by widening the band, and corrects a band that is off by a constant factor. It needs three
arrays and NumPy, nothing else. Any model, any framework, any number of input features.

```python
import uqcalibrate as uqc

report = uqc.evaluate(y_test, mean_test, std_test)
print(report)
```

The output below is real: 200 points from a seeded example whose model reports half the true
spread (`tests/conftest.py`, `honest_predictions`, seed 1, `scale=0.5`).

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

## Install

```
pip install uqcalibrate
```

Until the first release reaches PyPI, install from GitHub:
`pip install git+https://github.com/cgudumotu/uq`.

Python 3.10 or newer. The only dependency is NumPy.

## What you need

Three arrays, one value per test point:

| Argument | Meaning | Shape |
|---|---|---|
| `y` | the true values | `(n,)` |
| `mean` | the model's predictions | `(n,)` |
| `std` | the model's predictive standard deviation (the "give or take") | `(n,)` or one number |

If your model gives variances, pass `np.sqrt(var)`. If it gives several predictions per point
(an ensemble, Monte Carlo dropout, a Bayesian network), see *Combining samples* below.

## Reading the report

- **promised**: the level you asked for. "95%" means the band `mean ± 1.96 · std`.
- **delivered**: the share of true values that fell inside that band.
- **plausible range**: where the true coverage could be, given only `n` points (a 95% Wilson
  score interval). With 200 points an honest 95% band can deliver anywhere from about 92% to
  98% by chance alone. A wider range means weaker evidence.
- **verdict**: `OVERCONFIDENT` when even the top of the plausible range is below the promise
  (bands too narrow); `UNDERCONFIDENT` when even the bottom is above it (bands too wide);
  `CONSISTENT` when the promise lies inside the range, which means no evidence of a problem,
  not proof of calibration.
- **mean width**: the average width of the band. Narrower is better *only* when coverage holds.
- **interval score**: width plus a penalty for every miss (Gneiting & Raftery, 2007). Lower is
  better. Neither inflating nor shrinking the band improves it.
- **NLL**: the Gaussian negative log-likelihood, a "surprise score". Lower is better. A vague
  model is charged for being vague, so this cannot be gamed by widening the band.

The headline is the verdict at 95%. In code, `report.delivered` and `report.plausible_ranges`
hold the same numbers as fractions (0.65, not 65.0%), and `report.to_dict()` gives everything
as plain Python types.

## Fixing an overconfident model

If the bands are off by a constant factor, one number fixes them. Fit it on data the model has
**not** been evaluated on, then confirm on separate test data:

```python
s = uqc.fit_scaling(y_cal, mean_cal, std_cal)  # calibration split, not the test split
report = uqc.evaluate(y_test, mean_test, s * std_test)
print(report.headline)  # CONSISTENT: 95% promised, 94.0% delivered. No evidence of miscalibration.
```

`fit_scaling` cannot tell whether you reused the test split, so keep the splits separate.

## Combining samples

An ensemble, Monte Carlo dropout or a Bayesian network gives several predictions per point.
There is exactly one right way to turn them into one standard deviation, and it is the law of
total variance: add the average noise variance to the variance of the means, then take one
square root.

```python
# means, noise_vars: shape (S, n) — S members or samples, n points
std = uqc.total_std(means, noise_vars)
mean = means.mean(axis=0)
report = uqc.evaluate(y_test, mean, std)
```

Standard deviations are never added, and a square root is never taken of a standard deviation.
This package exists because a published thesis did both (see below).

## Comparing models

Several models, one test set: `compare` scores each with `evaluate`, ranks the ones that report
total uncertainty by their surprise score (never by coverage alone), and only names a winner
when the lead is bigger than its own margin of error. Models without a noise term (plain
Monte Carlo dropout, a Bayesian network without a variance output) go in `epistemic_only`: they
are shown, but not ranked against models that report total uncertainty.

```python
result = uqc.compare(
    y_test,
    {"deep ensemble": (mean_a, s_a * std_a), "single network": (mean_b, s_b * std_b)},
    epistemic_only={"MC dropout": (mean_c, std_c)},
)
print(result)
print(result.best)   # a name, or None when the top two are too close to call
```

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

That output is real (60 seeded points; the top two models differ only by a 0.9 factor on the
std, which 60 points cannot tell apart). Apply each model's own `fit_scaling` factor before
comparing, and compare on data no factor was fitted on.

## Many input features

The functions above never see your inputs `X`, so a model with 3 features or 300 works the same
way. Predict, then evaluate.

## Why this package exists

The author's master's thesis (*Uncertainty Quantification in Deep Neural Networks*, CSULB, 2024)
computed a deep operator neural network's total standard deviation as `√std(μ) + √mean(σ²)`
instead of `√(var(μ) + mean(σ²))`. The bands came out about 1.9× too wide, and the thesis's
headline result (89.05% coverage of a "95%" band) followed from that error. Coverage alone could
not catch it: wider bands always catch more points. [ERRATA.md](https://github.com/cgudumotu/uq/blob/main/ERRATA.md) lists every affected
table. It is a draft while the corrected numbers are regenerated; a `reproduce/` folder will
rebuild them with one command.

## API

| Function | Returns | Rule |
|---|---|---|
| `evaluate(y, mean, std, levels=(0.5, 0.8, 0.9, 0.95))` | `CalibrationReport` | MATH-4, MATH-7 |
| `fit_scaling(y, mean, std)` | `float` factor, > 0 | MATH-8 |
| `compare(y, predictions, epistemic_only=None, levels=...)` | `ModelComparison` | MODEL-5a, MODEL-3 |
| `total_std(means, noise_vars)` | `ndarray (n,)` | MATH-1 |
| `coverage(y, mean, std, level=0.95)` | `float` in [0, 1] | MATH-3 |
| `gaussian_nll(y, mean, std)` | `float` | MATH-5 |
| `interval_score(y, mean, std, level=0.95)` | `float` | MATH-6 |

Every rule is written out in [docs/constitution/MISSION.md](https://github.com/cgudumotu/uq/blob/main/docs/constitution/MISSION.md), and
every rule has a test that names it.

Coming in 0.2.0: `uqcalibrate[torch]`, with the five neural-network architectures from the
thesis and `train_and_compare(X, y)`, which trains them, calibrates each on a held-out split
and hands the results to `compare`.

## License

MIT. See [LICENSE](https://github.com/cgudumotu/uq/blob/main/LICENSE).
