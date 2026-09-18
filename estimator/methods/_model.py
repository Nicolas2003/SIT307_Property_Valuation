from __future__ import annotations

from ..contract import Estimate, Features, require
from ..ml import INPUT_COUNT, blank_count, fitted_models, to_frame

REQUIRED = ("SUBURB", "PROPERTY_TYPE", "LAND_SIZE", "NUM_BEDROOMS")


def estimate_with(model_key: str, features: Features) -> Estimate:
    missing = require(features, *REQUIRED)
    if missing:
        return Estimate.needs(*missing)

    price = fitted_models()[model_key].predict(to_frame(features))[0]
    blank = blank_count(features)
    note = (
        f"{blank} of {INPUT_COUNT} inputs left blank, filled from the training set."
        if blank
        else f"All {INPUT_COUNT} inputs answered."
    )
    return Estimate(price=float(price), notes=note)
