import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from response_service.api import create_app
from response_service.data import load, partitions
from response_service.drift import population_stability, reference_for
from response_service.schema import CATEGORICAL, EXCLUDED, FEATURES, NUMERIC, Customer
from response_service.scoring import Model
from response_service.train import reliability, train

FIXTURE = Path(__file__).parent / "fixtures" / "bank_sample.csv"


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("model")
    report = train(FIXTURE, root / "artifacts", root / "reports")
    path = root / "artifacts" / "model.json"
    sample = load(FIXTURE)[FEATURES].iloc[-1].to_dict()
    # Pandas mixes numpy integer and float scalars; JSON request should use native values.
    for name in ["age", "pdays", "previous"]:
        sample[name] = int(sample[name])
    for name in set(NUMERIC) - {"age", "pdays", "previous"}:
        sample[name] = float(sample[name])
    return path, report, sample


def test_partitions_are_contiguous_and_disjoint():
    sections = list(partitions(41188).values())
    assert sections[0].start == 0 and sections[-1].stop == 41188
    assert all(a.stop == b.start for a, b in zip(sections, sections[1:]))
    assert sum(s.stop - s.start for s in sections) == 41188


def test_leakage_columns_are_not_features(trained):
    path, report, _ = trained
    payload = json.loads(path.read_text())
    assert not set(EXCLUDED) & set(FEATURES)
    assert payload["training"]["excluded"] == EXCLUDED
    assert "y" not in FEATURES
    assert report["serving_max_abs_error"] < 1e-10


def test_preprocessing_is_fitted_only_on_training(trained):
    path, _, _ = trained
    payload = json.loads(path.read_text())
    frame = load(FIXTURE)
    training = frame.iloc[partitions(len(frame))["train"]]
    assert np.allclose(payload["means"], training[NUMERIC].mean())
    assert payload["categories"] == [sorted(training[c].unique()) for c in CATEGORICAL]


def test_threshold_is_selected_without_test_labels(tmp_path):
    baseline = train(FIXTURE, tmp_path / "first", tmp_path / "r1")
    frame = load(FIXTURE)
    frame = frame.rename(columns={"emp_var_rate": "emp.var.rate", "cons_price_idx": "cons.price.idx",
                                  "cons_conf_idx": "cons.conf.idx", "nr_employed": "nr.employed"})
    section = partitions(len(frame))["test"]
    frame.loc[frame.index[section], "y"] = frame.y.iloc[section].map({"yes": "no", "no": "yes"})
    changed = tmp_path / "changed.csv"
    frame.to_csv(changed, index=False, sep=";")
    result = train(changed, tmp_path / "second", tmp_path / "r2")
    assert result["threshold"] == pytest.approx(baseline["threshold"])
    one = json.loads((tmp_path / "first/model.json").read_text())
    two = json.loads((tmp_path / "second/model.json").read_text())
    for name in ["means", "scales", "categories", "coefficients", "intercept", "calibration"]:
        assert one[name] == two[name]


def test_inference_api_matches_exported_model(trained):
    path, _, sample = trained
    expected = Model.load(path).predict([sample])[0]
    with TestClient(create_app(path)) as client:
        assert client.get("/health").status_code == 200
        metadata = client.get("/model").json()
        response = client.post("/predict", json={"customers": [sample]}).json()
    assert response["model_version"] == metadata["model_version"]
    assert response["predictions"][0]["response_probability"] == pytest.approx(expected, abs=1e-12)
    assert 0 <= expected <= 1


@pytest.mark.parametrize("change", [{"duration": 120}, {"age": -1}, {"pdays": "999"},
                                    {"job": ""}, {"euribor3m": None}])
def test_invalid_api_features_are_rejected(trained, change):
    path, _, sample = trained
    with TestClient(create_app(path)) as client:
        response = client.post("/predict", json={"customers": [sample | change]})
    assert response.status_code == 422


def test_batch_bounds_and_extra_request_fields(trained):
    path, _, sample = trained
    with TestClient(create_app(path)) as client:
        for body in [{"customers": []}, {"customers": [sample] * 257},
                     {"customers": [sample], "target": "yes"}]:
            assert client.post("/predict", json=body).status_code == 422


def test_nonfinite_values_are_rejected(trained):
    _, _, sample = trained
    for value in [float("nan"), float("inf"), -float("inf")]:
        with pytest.raises(ValueError):
            Customer.model_validate(sample | {"euribor3m": value})


def test_unseen_categories_are_visible_and_score_is_finite(trained):
    path, _, sample = trained
    with TestClient(create_app(path)) as client:
        result = client.post("/predict", json={"customers": [sample | {"job": "new-role"}]}).json()
    assert result["unknown_category_counts"]["job"] == 1
    assert np.isfinite(result["predictions"][0]["response_probability"])


def test_missing_or_malformed_artifact_is_unavailable(tmp_path):
    for path in [tmp_path / "absent.json", tmp_path / "invalid.json"]:
        if path.name == "invalid.json":
            path.write_text('{"schema_version": 999}')
        with TestClient(create_app(path)) as client:
            assert client.get("/health").status_code == 503
            assert client.get("/model").status_code == 503


def test_model_rejects_invalid_dimensions_and_parameters(trained):
    path, _, _ = trained
    payload = json.loads(path.read_text())
    for change in [{"coefficients": [1]}, {"threshold": 2}, {"scales": [0] * len(NUMERIC)},
                   {"calibration": [float("nan"), 0]}]:
        with pytest.raises(ValueError):
            Model(payload | change, "test")


def test_drift_has_zero_for_reference_and_increases_for_shift():
    reference = reference_for([.5, 1.5])
    assert population_stability([.5, 1.5], reference) == pytest.approx(0)
    assert population_stability([1.5, 1.7], reference) > 1
    with pytest.raises(ValueError):
        population_stability([np.nan], reference)


def test_constant_training_feature_drift():
    reference = reference_for([0, 0])
    assert population_stability([0, 0], reference) == pytest.approx(0)
    assert population_stability([0, 1], reference) > 1


def test_two_distinct_values_and_out_of_support_shift_do_not_collapse():
    reference = reference_for([1, 1, 2, 2])
    assert population_stability([1, 1, 2, 2], reference) == pytest.approx(0)
    assert population_stability([2, 2, 2, 2], reference) > 1
    assert population_stability([9, 9], reference) > 10


def test_drift_accepts_small_unlabelled_batch(tmp_path):
    frame = load(FIXTURE).drop(columns="y").iloc[:2]
    path = tmp_path / "unlabelled.csv"
    frame.to_csv(path, index=False, sep=";")
    assert len(load(path, require_target=False)) == 2


def test_calibration_bins_account_for_every_boundary():
    probabilities = np.asarray([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
    rows = reliability(np.zeros(len(probabilities)), probabilities)
    assert sum(row["rows"] for row in rows) == len(probabilities)
    assert sum(row["rows"] for row in reliability(np.asarray([0, 1]), np.asarray([.3, .6]))) == 2


def test_training_metadata_is_required(trained):
    path, _, _ = trained
    payload = json.loads(path.read_text())
    del payload["training"]
    with pytest.raises(ValueError):
        Model(payload, "test")


@pytest.mark.parametrize("column,value", [("age", -10), ("euribor3m", float("inf")),
                                         ("job", ""), ("y", "maybe")])
def test_training_data_uses_serving_contract(tmp_path, column, value):
    frame = load(FIXTURE).rename(columns={"emp_var_rate": "emp.var.rate", "cons_price_idx": "cons.price.idx",
                                        "cons_conf_idx": "cons.conf.idx", "nr_employed": "nr.employed"})
    frame.loc[0, column] = value
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False, sep=";")
    with pytest.raises(ValueError):
        load(path)
