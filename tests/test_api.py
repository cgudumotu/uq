"""API-1 to API-6: the public surface, validation messages, naming, purity, dependencies, names."""

from __future__ import annotations

import ast
import inspect
import random
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import uqcalibrate as uqc
from conftest import honest_predictions

SRC = Path(uqc.__file__).resolve().parent
PUBLIC = {
    "CalibrationReport",
    "ModelComparison",
    "compare",
    "coverage",
    "evaluate",
    "fit_scaling",
    "gaussian_nll",
    "interval_score",
    "total_std",
}


# --- API-1 -------------------------------------------------------------------------------------


@pytest.mark.rule("API-1")
def test_api1_public_names_exact():
    import types

    # Importing a submodule binds it as an attribute of the package; those are not API.
    visible = {
        name
        for name in dir(uqc)
        if not name.startswith("_") and not isinstance(getattr(uqc, name), types.ModuleType)
    }
    assert visible == PUBLIC
    assert set(uqc.__all__) == PUBLIC


# --- API-2 -------------------------------------------------------------------------------------


@pytest.mark.rule("API-2")
def test_api2_std_not_positive_message():
    y = np.zeros(3)
    with pytest.raises(ValueError, match=r"^std has 2 values") as err:
        uqc.coverage(y, y, np.array([1.0, -0.2, 0.0]))
    assert str(err.value) == (
        "std has 2 values <= 0 (min -0.2). Standard deviations must be positive. "
        "If these are variances or log-variances, convert them first, e.g. std = np.sqrt(var)."
    )


@pytest.mark.rule("API-2")
def test_api2_samples_passed_as_mean_message(rng):
    y = np.zeros(200)
    means = rng.normal(size=(50, 200))
    with pytest.raises(ValueError, match=r"^mean has shape \(50, 200\)") as err:
        uqc.evaluate(y, means, 1.0)
    assert str(err.value) == (
        "mean has shape (50, 200) but y has shape (200,); mean must have shape (n,). "
        "If these are S samples per point, combine them first: mean = means.mean(axis=0); "
        "std = uqcalibrate.total_std(means, noise_vars)."
    )


@pytest.mark.rule("API-2")
def test_api2_column_vector_message():
    # A (n, 1) column is the shape PyTorch models usually return.
    y = np.zeros(4)
    with pytest.raises(ValueError, match=r"^mean has shape \(4, 1\)") as err:
        uqc.coverage(y, np.zeros((4, 1)), 1.0)
    assert str(err.value) == (
        "mean has shape (4, 1) but must have shape (n,): one value per point. "
        "Use mean.ravel() if it has one column."
    )


@pytest.mark.rule("API-2")
def test_api2_shape_mismatch_message():
    with pytest.raises(ValueError, match=r"^mean has shape \(199,\)") as err:
        uqc.gaussian_nll(np.zeros(200), np.zeros(199), 1.0)
    assert str(err.value) == (
        "mean has shape (199,) but y has shape (200,); they must match, one value per point."
    )


@pytest.mark.rule("API-2")
def test_api2_non_finite_message():
    y = np.array([0.0, 1.0, 2.0, np.nan, 4.0, np.inf])
    with pytest.raises(ValueError, match=r"^y has 2 non-finite values") as err:
        uqc.coverage(y, np.zeros(6), 1.0)
    assert str(err.value) == (
        "y has 2 non-finite values (NaN or inf), first at index 3. All inputs must be finite. "
        "Remove or fill those points before evaluating; uqcalibrate never drops them silently."
    )


@pytest.mark.rule("API-2")
def test_api2_empty_message():
    with pytest.raises(ValueError, match=r"^y is empty\. At least one point is required\.$"):
        uqc.coverage([], [], 1.0)


@pytest.mark.rule("API-2")
def test_api2_level_message():
    with pytest.raises(ValueError, match=r"^level must be strictly between") as err:
        uqc.coverage(np.zeros(2), np.zeros(2), 1.0, level=95)
    assert str(err.value) == (
        "level must be strictly between 0 and 1 (got 95). A 95% band is level=0.95."
    )


@pytest.mark.rule("API-2")
def test_api2_levels_messages(honest):
    y, mean, std = honest
    with pytest.raises(ValueError, match=r"^levels is empty\. Give at least one level"):
        uqc.evaluate(y, mean, std, levels=())
    with pytest.raises(
        ValueError, match=r"^levels has a duplicate \(0\.9\)\. Each level appears once\.$"
    ):
        uqc.evaluate(y, mean, std, levels=(0.9, 0.95, 0.9))
    with pytest.raises(ValueError, match=r"^levels must be a sequence of numbers"):
        uqc.evaluate(y, mean, std, levels=0.95)


@pytest.mark.rule("API-2")
def test_api2_means_not_2d_message():
    with pytest.raises(ValueError, match=r"^means has shape \(200,\)") as err:
        uqc.total_std(np.zeros(200), np.ones(200))
    assert str(err.value) == (
        "means has shape (200,) but must have shape (S, n): S samples or members on axis 0, "
        "n points on axis 1. For a single model use means[None, :], which gives S = 1."
    )


@pytest.mark.rule("API-2")
def test_api2_noise_vars_negative_message():
    with pytest.raises(ValueError, match=r"^noise_vars has 1 value < 0") as err:
        uqc.total_std([[0.0, 0.0]], [[1.0, -0.5]])
    assert str(err.value) == (
        "noise_vars has 1 value < 0 (min -0.5). Noise variances must be >= 0. "
        "If these are standard deviations, square them first: noise_vars = std**2."
    )


@pytest.mark.rule("API-2")
def test_api2_sample_shape_mismatch_message():
    with pytest.raises(ValueError, match=r"^noise_vars has shape \(2, 3\)") as err:
        uqc.total_std(np.zeros((2, 4)), np.ones((2, 3)))
    assert str(err.value) == (
        "noise_vars has shape (2, 3) but means has shape (2, 4); they must match."
    )


@pytest.mark.rule("API-2")
def test_api2_std_shape_message():
    with pytest.raises(ValueError, match=r"^std has shape \(3,\)") as err:
        uqc.coverage(np.zeros(200), np.zeros(200), np.ones(3))
    assert str(err.value) == (
        "std has shape (3,) but y has shape (200,); std must have shape (n,) or be one number."
    )


@pytest.mark.rule("API-2")
def test_api2_pending_messages(honest):
    # The one pending path left in the core (MATH-7, HANDOFF Q-1). P-04 was decided, so
    # fit_scaling no longer raises NotImplementedError.
    y, mean, std = honest
    with pytest.raises(NotImplementedError) as err:
        uqc.evaluate(y, mean, std, levels=(0.9,))
    assert str(err.value) == (
        "evaluate needs 0.95 among levels: the headline verdict is defined at 95% "
        "(MISSION MATH-7), and the headline for other levels is under review. Add 0.95 to levels."
    )


@pytest.mark.rule("API-2")
@pytest.mark.parametrize(
    "call",
    [
        lambda y, m, s: uqc.coverage(y, m, s),
        lambda y, m, s: uqc.gaussian_nll(y, m, s),
        lambda y, m, s: uqc.interval_score(y, m, s),
        lambda y, m, s: uqc.evaluate(y, m, s),
        lambda y, m, s: uqc.fit_scaling(y, m, s),
    ],
    ids=["coverage", "gaussian_nll", "interval_score", "evaluate", "fit_scaling"],
)
def test_api2_every_public_function_rejects_nan(call):
    y = np.array([0.0, np.nan])
    with pytest.raises(ValueError, match=r"non-finite"):
        call(y, np.zeros(2), np.ones(2))


@pytest.mark.rule("API-2")
def test_api2_inputs_are_converted_to_float64():
    out = uqc.total_std([[1, 2]], [[1, 1]])
    assert out.dtype == np.float64
    assert uqc.coverage([0, 1], [0, 0], 2) == 1.0


# --- API-3 -------------------------------------------------------------------------------------


@pytest.mark.rule("API-3")
def test_api3_parameter_names_carry_units():
    names = lambda f: list(inspect.signature(f).parameters)  # noqa: E731
    assert names(uqc.total_std) == ["means", "noise_vars"]
    assert names(uqc.coverage) == ["y", "mean", "std", "level"]
    assert names(uqc.gaussian_nll) == ["y", "mean", "std"]
    assert names(uqc.interval_score) == ["y", "mean", "std", "level"]
    assert names(uqc.evaluate) == ["y", "mean", "std", "levels"]
    assert names(uqc.fit_scaling) == ["y", "mean", "std"]
    assert names(uqc.compare) == ["y", "predictions", "epistemic_only", "levels"]
    # Nothing in the public surface is called `var` while holding a standard deviation.
    assert inspect.signature(uqc.coverage).parameters["level"].default == 0.95
    assert inspect.signature(uqc.evaluate).parameters["levels"].default == (0.5, 0.8, 0.9, 0.95)


# --- API-4 -------------------------------------------------------------------------------------


@pytest.mark.rule("API-4")
def test_api4_pure_no_output_no_random_state_change(capsys, honest):
    y, mean, std = honest
    np_state_before = np.random.get_state()  # noqa: NPY002  (the global state is the point)
    py_state_before = random.getstate()
    first = uqc.evaluate(y, mean, std)
    second = uqc.evaluate(y, mean, std)
    uqc.total_std([[1.0, 2.0], [3.0, 4.0]], [[1.0, 1.0], [1.0, 1.0]])
    assert first == second
    assert capsys.readouterr() == ("", "")
    np.testing.assert_equal(np.random.get_state(), np_state_before)  # noqa: NPY002
    assert random.getstate() == py_state_before


@pytest.mark.rule("API-4")
def test_api4_report_is_immutable_and_hashable(honest):
    import dataclasses

    y, mean, std = honest
    report = uqc.evaluate(y, mean, std)
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.nll = 0.0  # type: ignore[misc]
    assert isinstance(hash(report), int)


@pytest.mark.rule("API-4")
def test_api4_inputs_are_not_modified(rng):
    y, mean, std = honest_predictions(rng, 100)
    copies = (y.copy(), mean.copy(), std.copy())
    uqc.evaluate(y, mean, std)
    for original, copy in zip((y, mean, std), copies, strict=True):
        np.testing.assert_array_equal(original, copy)


@pytest.mark.rule("API-4")
def test_api4_to_dict_is_plain_python(honest):
    import json

    y, mean, std = honest
    d = uqc.evaluate(y, mean, std).to_dict()
    json.dumps(d)  # no custom encoder needed
    assert set(d) == {
        "n",
        "levels",
        "delivered",
        "plausible_ranges",
        "verdicts",
        "mean_widths",
        "interval_scores",
        "nll",
        "headline",
    }
    assert isinstance(d["n"], int)
    assert all(isinstance(v, float) for v in d["delivered"])


# --- API-5 -------------------------------------------------------------------------------------


@pytest.mark.rule("API-5")
def test_api5_imports_without_torch():
    code = "import sys; sys.modules['torch'] = None; import uqcalibrate; print('ok')"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "ok"


@pytest.mark.rule("API-5")
def test_api5_core_imports_only_numpy_and_stdlib():
    allowed = {"numpy"} | set(sys.stdlib_module_names)
    for path in SRC.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] in allowed, f"{path.name}: import {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:  # relative import: another core module
                    continue
                assert node.module.split(".")[0] in allowed, f"{path.name}: from {node.module}"


@pytest.mark.rule("API-5")
def test_api5_core_never_imports_reproduce_tests_or_models():
    forbidden = re.compile(
        r"^\s*(from|import)\s+(reproduce|tests|uqcalibrate\.models|\.models)\b", re.M
    )
    for path in SRC.glob("*.py"):
        assert not forbidden.search(path.read_text(encoding="utf-8")), path.name


# --- API-6 -------------------------------------------------------------------------------------


@pytest.mark.rule("API-6")
def test_api6_no_abbreviations_in_core_text():
    abbreviations = re.compile(r"\b(DONN|SNN|DENN|MCDO|MCDNN|BNN)\b")
    for path in SRC.glob("*.py"):
        assert not abbreviations.search(path.read_text(encoding="utf-8")), path.name


# --- API-2, the remaining validator branches -----------------------------------------------------


@pytest.mark.rule("API-2")
def test_api2_unconvertible_input_message():
    with pytest.raises(ValueError, match=r"^y could not be converted to numbers"):
        uqc.coverage(["a", "b"], [0.0, 0.0], 1.0)


@pytest.mark.rule("API-2")
def test_api2_three_dimensional_y_message():
    with pytest.raises(ValueError, match=r"^y has shape \(2, 2, 2\) but must have shape \(n,\)"):
        uqc.coverage(np.zeros((2, 2, 2)), np.zeros(8), 1.0)


@pytest.mark.rule("API-2")
def test_api2_noise_vars_not_2d_and_means_empty_messages():
    with pytest.raises(
        ValueError, match=r"^noise_vars has shape \(3,\) but must have shape \(S, n\)"
    ):
        uqc.total_std(np.zeros((1, 3)), np.ones(3))
    with pytest.raises(ValueError, match=r"^means is empty\."):
        uqc.total_std(np.zeros((0, 3)), np.ones((0, 3)))


@pytest.mark.rule("API-2")
def test_api2_level_not_a_number_message():
    with pytest.raises(
        ValueError, match=r"^level must be a number strictly between 0 and 1 \(got '0.95'\)"
    ):
        uqc.coverage(np.zeros(2), np.zeros(2), 1.0, level="0.95")
    with pytest.raises(
        ValueError, match=r"^level must be a number strictly between 0 and 1 \(got True\)"
    ):
        uqc.coverage(np.zeros(2), np.zeros(2), 1.0, level=True)


@pytest.mark.rule("API-2")
def test_api2_levels_zero_dimensional_array_is_rejected_with_value_error(honest):
    y, mean, std = honest
    with pytest.raises(ValueError, match=r"^levels must be a sequence of numbers"):
        uqc.evaluate(y, mean, std, levels=np.array(0.95))
    # A one-row (1, n) mean is a column of one sample, not "samples passed as a mean".
    with pytest.raises(ValueError, match=r"^mean has shape \(1, 2000\) but must have shape \(n,\)"):
        uqc.evaluate(y, mean[None, :], std)
