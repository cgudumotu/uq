"""uqcalibrate: check whether a regression model's uncertainty is honest, and fix it.

Three arrays in (true values, predicted means, predicted standard deviations), a verdict out::

    import uqcalibrate as uqc
    report = uqc.evaluate(y_test, mean_test, std_test)
    print(report)

The public surface is exactly the names in `__all__` (MISSION API-1). Everything else is
internal. The rules every function follows are in docs/constitution/MISSION.md.
"""

from .calibration import fit_scaling
from .combine import total_std
from .metrics import coverage, gaussian_nll, interval_score
from .report import CalibrationReport, evaluate

__version__ = "0.1.0"

__all__ = [
    "CalibrationReport",
    "coverage",
    "evaluate",
    "fit_scaling",
    "gaussian_nll",
    "interval_score",
    "total_std",
]
