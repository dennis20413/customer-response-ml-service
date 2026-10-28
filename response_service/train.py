import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             log_loss, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import download, load, partitions
from .drift import reference_for
from .schema import CATEGORICAL, EXCLUDED, FEATURES, NUMERIC
from .scoring import Model


def metrics(y, p, threshold):
    prediction = p >= threshold
    return {"rows": len(y), "positive_rate": float(np.mean(y)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)),
            "average_precision": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "precision": float(precision_score(y, prediction, zero_division=0)),
            "recall": float(recall_score(y, prediction, zero_division=0)),
            "f1": float(f1_score(y, prediction, zero_division=0))}


def reliability(y, p):
    bins = []
    edges = np.linspace(0, 1, 11)
    # Shared adjacent edges avoid floating-point gaps at 0.3, 0.6, etc.
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (p >= low) & ((p < high) if index < 9 else (p <= high))
        if mask.any():
            bins.append({"lower": float(low), "upper": float(high),
                         "rows": int(mask.sum()), "mean_probability": float(np.mean(p[mask])),
                         "observed_response": float(np.mean(y[mask]))})
    return bins


def train(csv, output=Path("artifacts"), reports=Path("reports")):
    frame = load(csv)
    split = partitions(len(frame))
    x = frame[FEATURES]
    y = (frame.y == "yes").astype(int).to_numpy()
    for name, section in split.items():
        if len(np.unique(y[section])) != 2:
            raise ValueError(f"Both target classes are required in {name}")
    prep = ColumnTransformer([("numeric", StandardScaler(), NUMERIC),
                              ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)])
    xt = prep.fit_transform(x.iloc[split["train"]])
    model = LogisticRegression(max_iter=2000, random_state=42).fit(xt, y[split["train"]])
    base = DummyClassifier(strategy="prior").fit(xt, y[split["train"]])
    cal = split["calibration"]
    calibrator = LogisticRegression(C=1e6, max_iter=2000, random_state=42).fit(
        model.decision_function(prep.transform(x.iloc[cal])).reshape(-1, 1), y[cal])
    validation = split["validation"]
    vp = calibrator.predict_proba(model.decision_function(prep.transform(x.iloc[validation])).reshape(-1, 1))[:, 1]
    precision, recall, thresholds = precision_recall_curve(y[validation], vp)
    scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    threshold = float(thresholds[int(np.argmax(scores))])
    scaler = prep.named_transformers_["numeric"]
    encoder = prep.named_transformers_["category"]
    payload = {"schema_version": 1, "numeric": NUMERIC, "categorical": CATEGORICAL,
               "means": scaler.mean_.tolist(), "scales": scaler.scale_.tolist(),
               "categories": [v.tolist() for v in encoder.categories_],
               "coefficients": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]),
               "calibration": [float(calibrator.coef_[0, 0]), float(calibrator.intercept_[0])],
               "threshold": threshold,
               "training": {"source_sha256": hashlib.sha256(Path(csv).read_bytes()).hexdigest(),
                            "rows": len(frame), "split": {k: [s.start, s.stop] for k, s in split.items()},
                            "python": platform.python_version(), "sklearn": sklearn.__version__,
                            "seed": 42, "excluded": EXCLUDED}}
    output, reports = Path(output), Path(reports)
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    model_path = output / "model.json"
    model_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    serving = Model.load(model_path)
    test = split["test"]
    records = x.iloc[test].to_dict(orient="records")
    p = serving.predict(records)
    expected = calibrator.predict_proba(model.decision_function(prep.transform(x.iloc[test])).reshape(-1, 1))[:, 1]
    if not np.allclose(p, expected, rtol=1e-10, atol=1e-12):
        raise AssertionError("JSON serving does not match trained sklearn pipeline")
    raw = serving.predict(records, calibrated=False)
    bp = base.predict_proba(prep.transform(x.iloc[test]))[:, 1]
    # Calibrator sees newer labels; compare against a prior with the same freshness.
    recent_prior = np.full(len(records), np.mean(y[cal]), dtype=float)
    slices = {}
    test_frame = frame.iloc[test]
    for name in ["contact", "housing"]:
        if name in test_frame:
            slices[name] = {}
            for group in sorted(test_frame[name].unique()):
                mask = (test_frame[name] == group).to_numpy()
                slices[name][group] = metrics(y[test][mask], p[mask], threshold)
    report = {"dataset": "UCI Bank Marketing bank-additional-full.csv", "model_version": serving.version,
              "protocol": "Preserved row order; train 50%, calibration 15%, threshold validation 15%, test 20%",
              "training": payload["training"], "threshold": threshold,
              "partition_positive_rates": {name: float(np.mean(y[section])) for name, section in split.items()},
              "validation_f1": float(np.max(scores)),
              "score_remapping": {"slope": payload["calibration"][0],
                                  "intercept": payload["calibration"][1],
                                  "reverses_ranking": payload["calibration"][0] < 0,
                                  "interpretation": "Logistic score remapping fitted only on calibration rows; a negative slope reverses ordering and is not monotonic probability calibration"},
              "test": {"train_prior_baseline": metrics(y[test], bp, threshold),
                       "calibration_period_prior_baseline": metrics(y[test], recent_prior, threshold),
                       "raw_logistic_regression": metrics(y[test], raw, threshold),
                       "score_remapped_logistic_regression": metrics(y[test], p, threshold)},
              "calibration_bins": reliability(y[test], p), "descriptive_slices": slices,
              "serving_max_abs_error": float(np.max(np.abs(p - expected))),
              "limitations": ["Old Portuguese campaign data; not evidence of Taiwan generalization",
                              "No customer IDs: recurring customers may cross time partitions",
                              "Row order is chronology proxy; exact timestamps unavailable",
                              "F1 threshold is a demonstration, not a verified business policy",
                              "Calibration can worsen under temporal distribution shift",
                              "Negative remapping slope reverses raw ranking: this is an observed shift failure, not proof of ordinary monotonic calibration"]}
    (reports / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    table = ["# Measured evaluation", "", report["protocol"], "", "| Test model | Log loss | Brier | AP | ROC AUC | F1 |", "|---|---:|---:|---:|---:|---:|"]
    for name, row in report["test"].items():
        table.append(f"| {name} | {row['log_loss']:.4f} | {row['brier']:.4f} | {row['average_precision']:.4f} | {row['roc_auc']:.4f} | {row['f1']:.4f} |")
    table.extend(["", f"Frozen validation threshold: `{threshold:.6f}`. Test rows: `{test.stop - test.start}`.",
                  "", f"Logistic score-remapping slope: `{payload['calibration'][0]:.6f}`; intercept: `{payload['calibration'][1]:.6f}`. A negative slope reverses raw ranking. This is not ordinary monotonic calibration.",
                  "", "Calibration and raw scores use the same validation-selected numeric threshold for transparent comparison; the raw model is not separately tuned.",
                  "", "".join(f"- {v}\n" for v in report["limitations"]),
                  "See evaluation.json for source SHA-256, partitions, versions, bins and descriptive slices."])
    (reports / "evaluation.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    # Persist a training-only numeric reference for an explicitly unlabelled drift diagnostic.
    reference = {}
    for name in NUMERIC:
        values = frame.iloc[split["train"]][name].to_numpy()
        reference[name] = reference_for(values)
    (output / "drift_reference.json").write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("data/bank-additional-full.csv"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    args = parser.parse_args()
    if args.download:
        download(args.csv.parent)
    report = train(args.csv, args.artifacts, args.reports)
    print(json.dumps({"model_version": report["model_version"], "test": report["test"]}, indent=2))


if __name__ == "__main__":
    main()
