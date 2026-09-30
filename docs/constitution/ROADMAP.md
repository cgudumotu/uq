# ROADMAP — uqcalibrate

**Status:** DRAFT constitution v1 · 2026-09-30 · amended the same day (Amendment log) · awaiting Carol's approval
**Legend:** ✅ done · 🔄 in progress · 📋 planned · ⏸ waiting on a decision (MISSION §9 or below)

Every item names the MISSION rules it serves. Update this file whenever an item changes state.

---

## Deadline (Carol, 2026-09-30)

M0, M1 and M2 finish before 2026-10-07. If anything slips, the PyPI upload slips first. Everything
else, including the TestPyPI upload, is finished by 2026-10-06; 2026-10-07 is buffer. The guest
lecture on 2026-10-12 does not depend on PyPI.

The pace is set by review: Claude drafts, and nothing counts as done until Carol has reviewed every
line. Each day's plan assumes the previous day's review is complete.

| Day | Milestone | Work |
|---|---|---|
| Wed 09-30 | M0, most of M1 | Decisions; constitution v1 placed; P-01 to P-03 decided; critique, UX copy, HANDOFF, system design, ADRs; scaffold; tests first; the whole NumPy core green; README v1 draft |
| Thu 10-01 | M0 done | Carol's line-by-line review of the core; P-04, P-05, P-07, P-08 decided; constitution v1 approved; `fit_scaling` formula lands (P-04) |
| Fri 10-02 | M1 | Review fixes; evidence manifest and provenance extraction start early |
| Sat 10-03 | M1 done | Evidence manifest, provenance extraction, stripped copies; DONN port; correction ladder; ERRATA E-1 numbers for approval |
| Sun 10-04 | M2 | Models (PyTorch extra); P-06, P-09 decided |
| Mon 10-05 | M2 | Corrected reruns of Tables 1–34 and figures; `compare()`; `train_and_compare()` |
| Tue 10-06 | M2 done | ERRATA complete for approval; CHANGELOG; 0.1.0 on TestPyPI (approval) |
| Wed 10-07 | Buffer | PyPI upload (approval); first item to slip |

**Critical path.** P-01 to P-05, P-07 and P-08 gate M1, so they are needed on 10-01. P-06 and
P-09 gate M2 only, so they can wait until 10-04. P-10 is decided (ADR-0002). Carol's TestPyPI and PyPI accounts exist
(2026-09-30), so the uploads wait only on her approval.

---

## M0 — Design (Phase 1: documents only, no code) · target Thu 2026-10-01

- ✅ Evidence verified against the notebooks (2026-09-30): σ formulas, data generators, seeds,
  saved outputs versus published tables. Results in ERRATA.md, Provenance.
- ✅ Constitution v1 drafted and placed in the uq folder: MISSION.md, TECH-STACK.md, ROADMAP.md.
  ⏸ Carol's approval.
- ✅ Carol's decisions of 2026-09-30 recorded: published computation for the DONN only (ADR-0001);
  deadline and slip order (above); no other copies of the notebooks (MISSION DATA-4).
- ✅ Local CLAUDE.md and `.gitignore` (TECH-STACK §7).
- 🔄 ERRATA.md drafted, numbers PENDING (REPRO-1, REPRO-2).
- ✅ P-01, P-02, P-03 decided (2026-09-30, as recommended); P-10 decided (ADR-0002).
  ⏸ P-04 to P-09 (MISSION §9); P-04, P-05, P-07, P-08 gate M1.
- ✅ Design critique and UX copy (docs/design/CRITIQUE.md, UX-COPY.md); MISSION amendments 7
  and 8 (report wording, `plausible_ranges`, plural field names).
- ✅ docs/design/HANDOFF.md: migration map, acceptance tests, decisions D-1 to D-4, open
  questions Q-1 to Q-4.

**Evidence (read-only, never edited):** Thesis_PyPi/DNN_Thesis_files/ (5 notebooks),
Thesis_PyPi/Ishigami_thesis_files/ (12 notebooks; the three `Ishigami_refactor_donn_x2_old`
copies are byte-identical, so one is used), Thesis_PyPi/UQ_DNN_Thesis.pdf.

## M1 — Prove the error, correct the headline, ship the NumPy core · target Sat 2026-10-03

- ✅ System design and architecture (docs/design/SYSTEM-DESIGN.md; ADR-0003 to ADR-0005;
  TECH-STACK amendment 2). 2026-09-30.
- ✅ Scaffold: uv project, src layout, pyproject.toml, LICENSE (MIT), uv.lock, CI workflow
  (TECH-STACK §2, §9). 2026-09-30. ⏸ Carol: the author line in pyproject and LICENSE.
- ✅ Core, tests first (MATH-1 to MATH-8, API-1 to API-6): 88 tests, seen failing first, then
  green at 100% line coverage (docs/design/TEST-LOG.md, CODE-REVIEW.md). 2026-09-30.
  `fit_scaling` raises `NotImplementedError` until P-04; its acceptance test is `xfail(strict)`.
- ✅ README v1 draft with a real example report. 2026-09-30. ⏸ Carol's line-by-line review of
  everything above.
- 📋 Evidence manifest, provenance extraction, stripped legacy copies (REPRO-5). 10-03.
- 📋 Published values transcribed from the PDF into reproduce/published_values.csv (REPRO-5d).
  10-03.
- 📋 reproduce/published/: faithful, seeded port of Refactor_donn; DEVIATIONS.md (REPRO-3,
  REPRO-4). 10-03.
- 📋 Reproduction of the published DONN numbers (REPRO-2 as decided). 10-03.
- 📋 Corrected DONN numbers along the correction ladder (ERRATA E-1). 10-03.
- 📋 ERRATA.md E-1 complete, other items PENDING. ⏸ Carol approves every number and claim. 10-03.

**Exit criteria:** MISSION §8 all checked.

## M2 — Models, corrected reruns, complete erratum, release · target Tue 2026-10-06

- 📋 `uqcalibrate.models` (PyTorch extra), all following MODEL-1 to MODEL-4: the simple, deep
  ensemble, Monte Carlo dropout, Bayesian and deep operator neural networks, named per API-6, with
  the thesis architectures unchanged (MODEL-6, ADR-0002). Open design questions go to ADRs: the
  Bayesian network implemented in-house or via blitz; how σ is formed for a model whose
  `noise_vars` is `None`. 10-04.
- 📋 `compare()` (MATH-4, MODEL-3). 10-05.
- 📋 `train_and_compare()` and `ModelComparison` (MODEL-5, MODEL-6; G6). ⏸ P-09. 10-05.
  Adds about half a day to M2; the PyPI upload remains the item that slips first.
- 📋 Corrected reruns of Tables 1–34 and the figures (REPRO-1). The published computation is
  re-run for the DONN only (REPRO-3, ADR-0001). ⏸ P-06 (DATA-4) for Table 34, whose notebook is
  lost. 10-05.
- 📋 ERRATA.md complete. ⏸ Carol approves every number and claim. 10-06.
- 📋 CHANGELOG, version 0.1.0. 10-06.
- ✅ GitHub repository `cgudumotu/uq` created, first commit pushed (2026-09-30). The repository
  name stays `uq`; the package name is `uqcalibrate` (Carol, 2026-09-30). The Trusted Publishing
  setup on PyPI names the repository, so the two names never need to match.
- ✅ TestPyPI and PyPI accounts created (Carol, 2026-09-30). PyPI requires two-factor
  authentication before any upload; the release checklist confirms it is switched on. The name
  `uqcalibrate` had no releases on PyPI or TestPyPI on 2026-09-30.
- 📋 TestPyPI upload. ⏸ Carol approves. 10-06.
- 📋 PyPI upload. ⏸ Carol approves. First item to slip. 10-07 or later.

## Parking lot (not scheduled)

CRPS (MATH-4 "where available"); conformal intervals; reliability diagrams and PIT histograms;
multi-output targets; non-Gaussian predictive distributions; unit-typed `Std` and `Variance`
wrappers that refuse to be added together; a documentation site (the README is the documentation
for 0.1.0); input-dependent scaling (a calibration factor that varies with x, for example
in-range versus out-of-range; MISSION §4, amendment 2); a noise head for the Bayesian network and
MC dropout, so that all five models could be ranked together (ADR-0002).

## Decision log

ADRs live in docs/decisions/.

- ADR-0001 (2026-09-30, accepted): the published computation is reproduced for the DONN only.
- ADR-0002 (2026-09-30, accepted): the built-in models keep the thesis architectures;
  `train_and_compare` ranks the three total-uncertainty models and shows two as epistemic-only.
- ADR-0003 (2026-09-30, accepted): validate once at the public boundary; private metric kernels.
- ADR-0004 (2026-09-30, accepted): normal quantiles from the standard library; Wilson intervals.
- ADR-0005 (2026-09-30, accepted): the NLL is computed through standardized residuals.

## Amendment log

- **2026-09-30** v1 drafted. Not yet approved.
- **2026-09-30, amendment 1.** Carol's decisions: deadline before 2026-10-07 with the PyPI upload
  as the first item to slip; published computation for the DONN only (ADR-0001). Added the day
  plan, the critical path and Carol's account prerequisites. Moved the documentation site from M2
  to the parking lot to fit the deadline. M2 reruns are now corrected-only except the DONN.
- **2026-09-30, amendment 2.** Carol created the TestPyPI and PyPI accounts and decided that the
  repository name stays `uq`. Both prerequisites are now marked done; the critical path no longer
  waits on them.
- **2026-09-30, amendment 3.** Input-dependent scaling added to the parking lot, following MISSION
  amendment 2 (multi-feature inputs are in scope; a per-region calibration factor is not, in v1).
- **2026-09-30, amendment 4.** `train_and_compare()` added to M2 (MISSION amendment 3, G6,
  MODEL-5, MODEL-6); P-09 and P-10 added to the 10-04 decisions.
- **2026-09-30, amendment 5.** P-10 decided (ADR-0002): removed from the 10-04 decisions; the
  noise-head question moved from the M2 ADR list to the parking lot; the `noise_vars is None`
  question added to the M2 ADR list.
- **2026-09-30, amendment 6.** Model names fixed by MISSION API-6; "a descriptive name for the
  DONN" removed from the M2 ADR list.
