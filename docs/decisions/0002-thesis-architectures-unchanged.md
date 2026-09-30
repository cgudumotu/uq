# ADR-0002: The built-in models keep the thesis architectures; three are ranked

- **Status:** Accepted
- **Date:** 2026-09-30
- **Decided by:** Carol
- **Rules affected:** MISSION G6, story 7, MODEL-5, MODEL-6 (P-10)

## Context

`train_and_compare(X, y)` (MODEL-5) trains the five thesis models and says which is best. A fair
ranking needs every ranked model to report the same kind of uncertainty. In the thesis, three
models output a mean and a noise variance (the single network, the deep ensemble and the DONN),
so their bands can hold both the noise term and the model-spread term. The Bayesian network and
MC dropout output a mean only, so their bands hold no noise term (ERRATA.md E-2). MISSION MODEL-3
forbids ranking those two against the other three.

Two options were on the table:

1. Add a `noise_head` switch to every model, on by default in `train_and_compare`, so all five
   report total uncertainty and can be ranked together; keep it off in the corrected reruns of the
   thesis tables so the reruns stay faithful.
2. Keep the thesis architectures everywhere. `train_and_compare` ranks the three total-uncertainty
   models and shows the other two in a separate epistemic-only section.

## Decision

Option 2. The package ships the five architectures as the thesis trained them. Nothing in the
one-call path is an architecture the thesis never tested.

## Consequences

- The ranked table has three rows; `.best` is always one of those three. The two epistemic-only
  models appear below it with the same columns (coverage, width, NLL, interval score) and a note
  that their bands hold no noise term, so MATH-4 is still met for every row.
- Users who want MC dropout or a Bayesian network *with* a noise term do not get one in v1. The
  noise head is a ROADMAP parking-lot item and can be added later by a new ADR, as a switch that
  defaults to off, without changing this decision's outputs.
- The corrected reruns and the one-call path use the same code for every model, so there is one
  implementation per architecture, not two.
- Open for the M2 architecture ADR: how σ is formed for a model whose `noise_vars` is `None`
  (MODEL-1), without changing the M1 signature of `total_std`.
