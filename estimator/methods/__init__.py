from __future__ import annotations

from ..contract import Method
from . import gradient_boosting, random_forest, svm

ESTIMATORS: list[Method] = [
    Method("random_forest", random_forest.LABEL, random_forest.BLURB, random_forest.estimate),
    Method("svm", svm.LABEL, svm.BLURB, svm.estimate),
    Method(
        "gradient_boosting",
        gradient_boosting.LABEL,
        gradient_boosting.BLURB,
        gradient_boosting.estimate,
    ),
]

__all__ = ["ESTIMATORS"]
