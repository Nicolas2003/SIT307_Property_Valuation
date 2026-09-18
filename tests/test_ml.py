from __future__ import annotations

import pytest

from estimator.csv_input import read_csv
from estimator.features import empty_features
from estimator.methods import ESTIMATORS
from estimator.ml.dataset import COLUMNS, to_frame, training_frame

SAMPLE = "data/sample_property.csv"


@pytest.fixture(scope="module")
def sample_features():
    with open(SAMPLE, "rb") as fh:
        upload = read_csv(fh)
    assert upload.ok, [str(e) for e in upload.report.errors]
    return upload.features


def test_blank_row_matches_the_fit_time_schema():
    frame = to_frame(empty_features())
    training_inputs, _ = training_frame()

    assert list(frame.columns) == COLUMNS
    assert frame.dtypes.equals(training_inputs.dtypes)

    numeric = training_inputs.select_dtypes(include=["number"]).columns
    assert frame[numeric].isna().all(axis=None)


def test_sample_property_round_trips(sample_features):
    frame = to_frame(sample_features)

    assert frame.loc[0, "LAND_SIZE"] == pytest.approx(1407.0)
    assert frame.loc[0, "SUBURB"] == "Mosman"
    assert frame.loc[0, "POSTCODE"] == pytest.approx(2088.0)

    assert frame.loc[0, "LAST_SALE_PRICE"] == pytest.approx(520000.0)
    assert frame.loc[0, "PRICE_GROWTH"] == pytest.approx(99500.0)


@pytest.mark.parametrize("method", ESTIMATORS, ids=lambda m: m.key)
def test_method_prices_the_sample(method, sample_features):
    estimate = method.estimate(sample_features)

    assert estimate.price is not None, estimate.notes
    assert 100_000 < estimate.price < 20_000_000
    assert estimate.notes


@pytest.mark.parametrize("method", ESTIMATORS, ids=lambda m: m.key)
def test_method_declines_without_a_suburb(method, sample_features):
    estimate = method.estimate(dict(sample_features, SUBURB=None))

    assert estimate.price is None
    assert "SUBURB" in estimate.notes
