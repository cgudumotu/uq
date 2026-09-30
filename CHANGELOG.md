# Changelog

All notable changes to uqcalibrate are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html): while the version starts with 0, a
minor release (0.2.0) may still change the public API.

## [Unreleased]

## [0.1.0] - Unreleased

The NumPy core. Three arrays in (true values, predicted means, predicted standard deviations), a
verdict out. NumPy is the only dependency.

### Added

- `evaluate`: promised versus delivered coverage at 50, 80, 90 and 95%, each with a 95% Wilson
  interval ("plausible range") and a verdict (OVERCONFIDENT, UNDERCONFIDENT or CONSISTENT), plus
  mean band width, interval score and Gaussian NLL, in an immutable `CalibrationReport`.
- `fit_scaling`: one factor that fixes bands off by a constant, fitted on held-out data (the
  closed-form NLL minimizer).
- `compare` and `ModelComparison`: ranks several models by NLL on the same test data, shows
  coverage and width beside it, and names a winner only when its lead beats two standard errors;
  models without a noise term are shown separately and never ranked.
- `total_std`: combines ensemble members or Monte Carlo samples by the law of total variance.
- `coverage`, `gaussian_nll`, `interval_score`: the individual scores.
- Error messages that name the argument, the problem and the fix for every invalid input.

[Unreleased]: https://github.com/cgudumotu/uq/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/cgudumotu/uq/releases/tag/v0.1.0
