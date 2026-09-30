# Design critique: the user-facing surface of uqcalibrate

**Phase 1, step 2 · 2026-09-30 · reviewed:** README.md (draft), MISSION §6 (names, signatures,
`CalibrationReport`), the report wording, and the error-message contract (API-2).
**Stage:** refinement. The math is fixed by MISSION §7; only names, wording and layout are open.
**Method:** the design-critique framework (first impression, usability, hierarchy, consistency,
accessibility), applied to a command-line and API surface instead of a screen.

## Overall impression

The product is the printed report, and the README puts it on the first screen: a user sees in
ten seconds what they will get. The biggest opportunity is the moment things go wrong: the two
most likely user mistakes (passing `(S, n)` samples where a `(n,)` mean belongs, and passing
variances as `std`) are caught by API-2, but the wording that guides the user back was not yet
designed. The second opportunity is the report's own robustness: it used characters that break
on a Windows console when output is redirected to a file.

## Usability

| Finding | Severity | Recommendation |
|---|---|---|
| `mean` (shape `(n,)`) and `means` (shape `(S, n)`) differ by one letter. A user with an ensemble will pass `means` to `evaluate` and get a shape error. | 🟡 Moderate | Keep the names (the plural is the right signal) and make the shape error teach: name both shapes and show the two-line fix (`mean = means.mean(axis=0)`, `std = total_std(...)`). Added to API-2 as a second example. |
| `fit_scaling` returns a factor the user must apply as `s * std`. A user may expect a corrected array back. | 🟢 Minor | Keep the factor: it is transparent (the user sees the number, can log it, can apply it to new data). Every example shows the multiplication. |
| `evaluate` with `levels` that omit 0.95 raises `NotImplementedError` (MATH-7, headline level undecided). Without guidance this looks like a bug. | 🟡 Moderate | The message says what to do: "include 0.95 in levels, or wait for the MATH-7 decision". |
| Report fields are fractions (`delivered[3] == 0.65`) while the printed report shows percentages (65.0%). | 🟢 Minor | Correct by MATH-3; document once in the `CalibrationReport` docstring and in the README's "Reading the report". |
| The report showed the verdict but not the evidence behind it, so `CONSISTENT` at n = 10 looked as strong as `CONSISTENT` at n = 100,000 (edge case E-c). | 🔴 Critical | Show the Wilson interval beside the delivered coverage in every row, and add one fixed footer line explaining that a wider range means weaker evidence. No threshold to invent. |
| Users must compute `mean = means.mean(axis=0)` themselves after `total_std`. | 🟢 Minor | Acceptable in v1 (single responsibility: `total_std` computes σ and nothing else). A `combine()` returning both is a parking-lot candidate, not an M1 change. |

## Reading order (hierarchy)

- **What draws the eye first:** the printed report block under the three-line example. Correct:
  that block *is* the product.
- **Reading flow of the README:** tagline → example → output → install → inputs → reading the
  report → fixing → combining → many features → why → API → license. Each section answers the
  question the previous one raises. Kept.
- **Reading flow of the report:** header (n, band type) → one row per level, promised before
  delivered → NLL → headline. The headline is last so it is the final thing on screen in a
  terminal, which is where the eye lands. Kept.
- **Emphasis:** verdicts in capitals are the only shouting in the report, and they are the thing
  a user acts on. Kept.

## Consistency

| Element | Issue | Recommendation |
|---|---|---|
| Column header "95% margin" | Sits next to a "95%" level row and reads as "the margin at the 95% level". | Rename to "plausible range" and define it once in the footer line ("95% Wilson interval for the true coverage"). |
| Argument order | `(y, mean, std)` everywhere, `y` first, matching the `(y_true, y_pred)` convention of scikit-learn metrics. | Kept. |
| Units in names (API-3) | `std`, `noise_vars`, `mean_width`: each name says what it holds. | Kept. |
| `CalibrationReport` fields | The printed report shows the Wilson interval, but §6 had no field for it, so `to_dict()` would lose the evidence. | Add `plausible_ranges`: one `(low, high)` pair per level. |
| Verdict words | OVERCONFIDENT / UNDERCONFIDENT / CONSISTENT. "CONSISTENT" can be read as "calibrated". | Kept, because the alternatives ("PASS", "OK") read even more like proof. The footer line and the README define it as "no evidence of a problem". |
| Report field names (found in the architecture step) | `mean_width` and `interval_score` hold one value per level; `delivered`, `verdicts`, `plausible_ranges` are plural. | Renamed `mean_widths`, `interval_scores` (MISSION amendment 8, for Carol's veto). |

## Accessibility

- **Plain text:** the report is a fixed-width table with one row per level; `to_dict()` gives
  the same numbers as plain Python types for screen readers, notebooks and JSON.
- **Character set:** the draft used an en dash and could have used σ or ±. On Windows, Python
  prints UTF-8 to a real console, but output redirected to a file uses the system code page
  (cp1252), where σ raises `UnicodeEncodeError`. **`str(report)` is ASCII only:** hyphen for
  ranges, `std` for σ, `+/-` never needed.
- **Width:** the widest row is under 80 characters, so nothing wraps in a default terminal.
- **Contrast and touch targets:** not applicable.

## What works well

- The three-array contract (`y`, `mean`, `std`) means a user needs no adapter for any framework.
- Every metric in the report is defined in one sentence in the README, and each definition says
  which direction is better.
- The "Why this package exists" section is honest about the thesis error without dramatizing it,
  and it points at the erratum and the one-command reproduction.
- The rule IDs in the API table let a reviewer go from a function to its rule in one hop.

## Priority recommendations (applied; see UX-COPY.md)

1. **Show the evidence in every row and keep the report ASCII.** Add the "plausible range"
   column and the footer line; drop non-ASCII characters from `str(report)`. Amends MATH-7
   wording and the §6 illustration.
2. **Add `plausible_ranges` to `CalibrationReport`** so `to_dict()` carries the evidence.
   Amends §6.
3. **Make the two common mistakes teach.** Add the `(S, n)`-for-`(n,)` example to API-2, next to
   the variances-for-std example, and fix the error-message template.

## Not changed, and why

- Function names: all six are kept. `evaluate` is generic, but it is the verb Carol used in the
  seed prompt and the one a new user would guess. `gaussian_nll` is jargon, but it is the name
  every paper uses, and the README explains it as a surprise score.
- The example-first README layout: the output block is the fastest way to explain the product.
