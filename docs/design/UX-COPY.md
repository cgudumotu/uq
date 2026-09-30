# UX copy: the printed report and the error messages

**Phase 1, step 2 · 2026-09-30 · applies** the three priority recommendations in CRITIQUE.md.
**Context:** a data scientist runs `print(uqc.evaluate(y, mean, std))` in a terminal or a
notebook, or hits a `ValueError` on the first call. **Tone:** plain, factual, never alarmed;
errors are written as a helpful colleague would say them. **Constraints:** ASCII only; every
row under 80 characters; the same term for the same thing everywhere (README, report, docstrings,
errors).

## Recommended copy

**Report header**
`uqcalibrate: 200 points, Gaussian bands`

**Column headers** (fixed width, 79 characters; the verdict column is 14 wide for UNDERCONFIDENT)
`promised  delivered  plausible range  verdict        mean width  interval score`

**A row**
`     95%      65.0%    58.2% - 71.3%  OVERCONFIDENT        2.35           10.88`

**NLL line**
`NLL 2.39`

**Footer** (always printed; the only place the range is defined; two lines so each stays under 80 columns)
`plausible range: 95% Wilson interval for the true coverage.`
`A wider range means weaker evidence.`

**Headline, one of three**
- `OVERCONFIDENT: 95% promised, 65.0% delivered.`
- `UNDERCONFIDENT: 95% promised, 99.5% delivered.`
- `CONSISTENT: 95% promised, 94.0% delivered. No evidence of miscalibration.`

**Complete example** (a real run: README, seed 1, model reporting half the true spread)

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

**Error messages (API-2).** Template: *what is wrong, with the number that proves it* +
*what is required* + *the likely cause and the fix, as code*. One sentence each, in that order.

| Situation | Message |
|---|---|
| `std` not positive | `std has 3 values <= 0 (min -0.2). Standard deviations must be positive. If these are variances or log-variances, convert them first, e.g. std = np.sqrt(var).` |
| samples passed as a mean | `mean has shape (50, 200) but y has shape (200,); mean must have shape (n,). If these are S samples per point, combine them first: mean = means.mean(axis=0); std = uqcalibrate.total_std(means, noise_vars).` |
| shapes disagree | `mean has shape (199,) but y has shape (200,); they must match, one value per point.` |
| non-finite values | `y has 2 non-finite values (NaN or inf), first at index 3. All inputs must be finite. Remove or fill those points before evaluating; uqcalibrate never drops them silently.` |
| empty input | `y is empty. At least one point is required.` |
| `level` out of range | `level must be strictly between 0 and 1 (got 95). A 95% band is level=0.95.` |
| `means` not 2-D | `means has shape (200,) but must have shape (S, n): S samples or members on axis 0, n points on axis 1. For a single model use means[None, :], which gives S = 1.` |
| `noise_vars` negative | `noise_vars has 1 value < 0 (min -0.5). Noise variances must be >= 0. If these are standard deviations, square them first: noise_vars = std**2.` |
| a pending rule (used by `fit_scaling` until P-04 was decided) | `<function> is not available yet: <what> (MISSION <rule>, decision <P-id>) is under review. See docs/constitution/MISSION.md, section 9.` |
| headline level missing | `evaluate needs 0.95 among levels: the headline verdict is defined at 95% (MISSION MATH-7), and the headline for other levels is under review. Add 0.95 to levels.` |

## Addendum, P-09: the comparison table (`compare`)

**Header:** `uqcalibrate: 3 models ranked, 1 epistemic-only, 60 points`
**Columns:** `rank  model  95% delivered  mean width  NLL  verdict`; the model column is as wide
as the longest name; rows are at most 100 columns with the thesis names (they cannot be under
80 and still hold "Monte Carlo dropout neural network").
**Headlines, one of four:**
- `Best: <name> (leads the runner-up by 0.06 +/- 0.02 NLL per point).`
- `Too close to call: the top two differ by 0.00 +/- 0.02 NLL per point.` (the runner-up is not
  named, so the line stays short; the table shows who ranks second)
- `Best: <name> (the only model ranked).`
- `No model reports total uncertainty; nothing ranked.`
**Second section, only when needed:** a blank line, then
`Epistemic only (bands hold no noise term; not ranked)`, with `-` in the rank column.

| Situation | Message |
|---|---|
| `predictions` not a mapping | `predictions must be a mapping from model name to (mean, std), e.g. {'my model': (mean, std)} (got list).` |
| a value is not a pair | `predictions['a'] must be a pair (mean, std) (got 3 items).` |
| nothing to compare | `compare needs at least one model: both predictions and epistemic_only are empty.` |
| bad name | `model names must be non-empty strings (got 3).` |
| a name in both mappings | `'a' appears in both predictions and epistemic_only; a model is one or the other.` |
| one point | `compare needs at least 2 points to estimate the lead's standard error (got 1).` |

## Alternatives considered

| Option | Copy | Tone | Best for |
|---|---|---|---|
| A (chosen) | "plausible range" + footer definition | plain | first-time readers; no double "95%" in the row |
| B | "95% CI" column | technical | statisticians; rejected because "95% CI" beside the "95%" level row reads as one thing |
| C | "margin +/- 6.1" | compact | rejected: Wilson intervals are not symmetric, so a single +/- number would be wrong |
| Headline, consistent | "CONSISTENT: ... No evidence of miscalibration." (chosen) | honest | avoids reading CONSISTENT as a pass mark |
| Headline, consistent | "CALIBRATED: ..." | reassuring | rejected: claims proof the data cannot give |

## Rationale

- The footer line defines the one term a newcomer will not know, in the place they see it, and
  states the reading rule ("wider means weaker evidence") instead of a threshold, so edge case
  E-c is handled without inventing a cut-off.
- Errors lead with the number that proves the problem (`3 values <= 0 (min -0.2)`), so the user
  can confirm it against their own data, then give the fix as code they can paste.
- The two most likely mistakes (samples passed as a mean; variances passed as standard
  deviations) get their own messages, because a generic shape error would send the user to the
  docs.
- ASCII throughout: the hyphen in `67.3% - 79.4%`, `std` rather than the Greek letter, `<=`
  rather than the symbol. The report prints the same on every console and in every log file.

## Localization notes

Not localized in v1. Decimal point is `.`; percentages have one decimal; the column widths are
fixed, so translated headers must be no wider than the English ones. Verdict words are
identifiers as well as display text and must not be translated.
