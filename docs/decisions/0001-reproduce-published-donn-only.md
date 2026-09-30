# ADR-0001: Reproduce the published computation for the DONN only

- **Status:** Accepted
- **Date:** 2026-09-30
- **Decided by:** Carol
- **Rules affected:** MISSION G4, story 5, DATA-5, REPRO-1 to REPRO-4; TECH-STACK §3, §4

## Context

The erratum needs two kinds of numbers. Published numbers, re-run from the published code, show
that the error reproduces. Corrected numbers replace them.

- The headline error, the DONN's total standard deviation (ERRATA.md E-1), lives in one notebook,
  `Refactor_donn.ipynb`.
- The other four models' published σ definitions can be stated exactly from their code (ERRATA.md
  E-2). Re-running them would reproduce the published numbers again; it would not change any
  corrected number.
- Faithful ports of the other four notebooks would add four ports, their seeded reruns, and a
  dependency used only by the published BNN (`blitz-bayesian-pytorch`), before a 2026-10-07
  deadline.
- The notebook behind the Ishigami table (Table 34) is lost, so its published computation cannot be
  ported at all.

## Decision

reproduce/published/ contains a faithful, seeded port of `Refactor_donn.ipynb` and nothing else.
Every table, the DONN's included, is regenerated with the corrected computation. The other models'
published numbers are cited from the PDF with their provenance (REPRO-5d), not regenerated.

## Consequences

- The headline claim is tested directly: rung 1 of the correction ladder re-runs the published
  DONN code, and rung 2 changes only the σ formula (ERRATA.md E-1).
- For the other models, the erratum shows the published σ definition from the code and the
  corrected numbers, but no seeded re-run of the published code.
- If P-07 is accepted as proposed, REPRO-2b (statistical agreement) applies to the DONN only, and
  REPRO-2a (arithmetic checks) still covers every published table.
- The reproduce dependency group drops `blitz-bayesian-pytorch`. It keeps pandas and seaborn,
  which `Refactor_donn.ipynb` uses for its results table and its band plot.
- Reversible: another faithful port can be added later by a new ADR that supersedes this one. The
  package itself is unaffected either way.
