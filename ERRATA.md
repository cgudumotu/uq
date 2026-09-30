# Errata — *Uncertainty Quantification in Deep Neural Networks*

M.S. thesis, Carol Eunice Gudumotu, California State University, Long Beach, May 2024.

**Status: DRAFT, 2026-09-30.** Every corrected number is **PENDING** until the reproduction in
reproduce/ runs. Nothing here is final until Carol approves it. Items E-1 to E-8 are verified
against the original notebooks and the thesis PDF (2026-09-30); corrected values arrive in M1 (E-1)
and M2 (the rest).

---

## Summary

The thesis's headline result is that the deep operator neural network (DONN) captures 89.05% of
test points inside its "95%" band, far above the other four models (Table 32; also the abstract,
p. ii, and the conclusion, p. 51). That result comes from an error in how the DONN's total
predictive standard deviation is computed (Figure 16, p. 24), not from the architecture. Corrected
value: **PENDING**. Related problems in the comparison and the data are listed below; each will be
regenerated and published here.

This errata does not claim that any of the five architectures is worse than published. It states
that the published evidence does not support the published ranking. The corrected numbers will
state what the evidence does support.

---

## E-1 The DONN's total predictive standard deviation (headline) · M1

**Where:** Figure 16 (p. 24); `Refactor_donn.ipynb`, code cell 9 (0-based index), function
`plot_with_confidence_interval`.

**Published:**

```python
stds = torch.stack(mus).std(axis=0).detach().numpy()**(1/2) + \
       torch.stack(vars).mean(axis=0).detach().numpy()**(1/2)
```

that is, σ = √std(μ) + √mean(σ²), over 50 stochastic passes.

**Correct** (law of total variance; uqcalibrate MISSION MATH-1): σ = √( var(μ) + mean(σ²) ).

**Two independent defects:**
1. A square root applied to a standard deviation. `std` already took one; the result has the
   units of √y (if y is in dollars, √dollars), which cannot be added to a quantity in dollars.
2. Standard deviations added instead of variances. Independent spreads combine like the sides of
   a right triangle: √(a² + b²), not a + b.

**Effect:** bands inflated by about 1.9× on average (**PENDING**: exact figure from reproduce/).

**Affected:** Tables 2, 7, 12, 17, 22, 27 and 32; Figures 28b, 29b, 30b; abstract; conclusion.

**Correction ladder.** Each rung changes exactly one thing, so each change's effect is visible.

| Rung | Change | Coverage, full test | In range | Out of range | 95% width | NLL |
|---|---|---|---|---|---|---|
| 0 | Published (Table 32) | 89.05 | 85.42 | 97.00 | not reported | not reported in this form |
| 1 | Published code, re-run with seeds | PENDING | PENDING | PENDING | PENDING | PENDING |
| 2 | + σ by the law of total variance | PENDING | PENDING | PENDING | PENDING | PENDING |
| 3 | + "95%" band = ±1.96σ (was ±2σ) | PENDING | PENDING | PENDING | PENDING | PENDING |
| 4 | + test noise from the training law, independent stream (E-3) | PENDING | PENDING | PENDING | PENDING | PENDING |

For comparison, the Simple NN (Tables 1 and 32): 63.45 / 65.92 / 47.99. It was computed on the same
data with the same ±2σ bands, so rung 2 against the Simple NN is a like-for-like comparison.

## E-2 Five models, five different uncertainty definitions (Table 32) · M2

Table 32 puts five quantities in one column:

| Model | σ as computed | Notebook cell |
|---|---|---|
| SNN | √var (noise only; a single network has no model spread) | `Refactor_snn` 9 |
| DONN | √std(μ) + √mean(var) (E-1) | `Refactor_donn` 9 |
| DENN | √std(μ): noise variances are collected but never used, plus the extra square root | `Refactor_denn` 10 |
| BNN | spread of sampled predictions only; no noise term | `Refactor_bnn` 11 |
| MCDO | spread of sampled predictions only; no noise term | `Refactor_mcdnn` 10 |

Only the DONN's band contains both a noise term and a model-spread term, so it wins the coverage
column by construction. BNN and MCDO are epistemic-only; under MISSION MODEL-3 they are reported
separately, not ranked against the others. Corrected Table 32: **PENDING** (M2).

## E-3 Test noise differs from training noise (Tables 1–33) · M2; DONN in M1 (rung 4)

`sample_dataset` computes the noise standard deviation from its range arguments,
|(|start| + |end|)/2 − |x|| / 16 (`Refactor_donn` cell 3, same in all five notebooks). The test set,
`sample_dataset(-10, 10, 200)` (cell 4), therefore has larger noise than training (range −7 to 7)
at every shared x. Both sets use `random_state=0`, so the test noise is the training noise stream
rescaled. The "in-training-distribution" subset is not drawn from the training distribution.

## E-4 "95%" bands were ±2σ (all coverage tables) · M2; DONN in M1 (rung 3)

All five notebooks build the band as μ ± 2σ, which is a 95.45% band. A 95% band is μ ± 1.96σ.

## E-5 The Ishigami experiment (Table 34; Figures 10–12, 17, 18, 31) · M2

**As printed:** the generator in Figure 17 sorts x1, x2 and x3 separately
(`np.sort(np.random.uniform(...))`, p. 25). Sorting makes the three inputs perfectly rank-correlated,
so the "three-input" experiment is effectively one-dimensional.

**Provenance (see below):** the notebook that produced Table 34 and Figures 10, 17 and 18 is not
among the surviving files. Corrected Table 34: **PENDING** (M2; needs MISSION decision DATA-4).

## E-6 Claims in the abstract and conclusion · M2

- *"a pioneering deep learning-based framework termed Bayesian deep operator neural networks"*
  (abstract): the DONN code contains no prior, posterior, variational or KL component.
- *"surpasses existing models"* (abstract) and *"state-of-the-art deep operator neural network
  model was able to capture and estimate aleatoric and epistemic uncertainty"* (conclusion):
  superseded pending E-1 to E-5.

Replacement wording: **PENDING**, written after the corrected numbers exist.

## E-7 Table 3 does not report a test error for MC dropout · M2

`Refactor_mcdnn` cell 11 appends the last *training* loss (`all_test_losses.append(loss.item())`)
where a test loss belongs, so Table 3's "MSE Testing" (0.08 ± 0.04) is not a test error. Tables 31
and 33 report 0.198 ± 0.034 for the same quantity. Corrected value: **PENDING** (M2).

## E-8 Table 31 counts one ensemble member, not five · M2

Table 31 lists the deep ensemble at 3,902 parameters to support "we have ensured that the number of
parameters in the architecture is almost equal." `Refactor_denn` trains `num_models = 5` networks of
hidden size 60; one such network has 3,902 parameters, so the ensemble has 19,510.

---

## Provenance of the published numbers (verified 2026-09-30)

| Model | Published tables | Saved notebook outputs | Source of the published numbers |
|---|---|---|---|
| SNN | 1, 6, 11, 16, 21, 26 | Match exactly: all three test sets, run by run | Notebook + PDF |
| DENN | 5, 10, 15, 20, 25, 30 | Match exactly: all three test sets, run by run | Notebook + PDF |
| BNN | 4, 9, 13, 19, 23, 29 | A later re-run (full-test mean 13.4; published 13.3) | PDF only |
| MCDO | 3, 8, 14, 18, 24, 28 | A later re-run (full-test mean 52.6; published 52.15) | PDF only |
| DONN | 2, 7, 12, 17, 22, 27 | Two runs of an interrupted re-run (90.0, 89.0) | PDF only |
| Ishigami DONN | 34 | No saved output matches (e.g. `donn_x1`: 86.9 / 70.2 / 100.0) | PDF only |
| All | 31, 32, 33 | Compiled by hand from the per-model tables; no notebook counts parameters (Table 31) | PDF only |

**The surviving Ishigami notebooks are a different experiment from the printed one.** All 11
unique notebooks draw inputs without sorting, use a constant noise σ = 0.2 (the printed generator
uses 0.2·|f(x)|), train on ±π/2 (the thesis states ±2π/3), and use DONN hidden size 35 (Figure 18
shows 10). Figure 10's scatter shows the signature of sorted inputs: y is a thin curve in x1, which
these notebooks cannot produce. The three `Ishigami_refactor_donn_x2_old` copies are
byte-identical.

**No notebook seeds PyTorch**, so no published run can be replayed exactly. Reproductions are
statistical (MISSION REPRO-2).

---

## Under verification (not yet claims)

Items from an earlier review, each to be checked against the notebooks before it can enter the
errata: training loss and MSE are reported from the last mini-batch only; test metrics come from
epoch 140 while the plotted model is from epoch 149; the deep ensemble's reported test MSE comes
from one member, not the ensemble.

## How to reproduce

**PENDING** (M1): one command, CPU only, fixed seeds. See MISSION REPRO-1.
