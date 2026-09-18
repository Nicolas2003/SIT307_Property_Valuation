from __future__ import annotations

from ..contract import Estimate, Features
from ._model import estimate_with

LABEL = "Gradient Boosting"
BLURB = "400 shallow trees fitted in sequence, each correcting the last."


def estimate(features: Features) -> Estimate:
    return estimate_with("gradient_boosting", features)
