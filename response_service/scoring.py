"""Safe JSON model serving: no executable pickle deserialization."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .schema import CATEGORICAL, NUMERIC


def sigmoid(values):
    values = np.clip(np.asarray(values, dtype=float), -700, 700)
    return 1.0 / (1.0 + np.exp(-values))


class Model:
    def __init__(self, payload: dict, version: str):
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("Unsupported model schema")
        if payload.get("numeric") != NUMERIC or payload.get("categorical") != CATEGORICAL:
            raise ValueError("Model feature contract does not match application")
        if not isinstance(payload.get("training"), dict):
            raise ValueError("Required training metadata is missing")
        self.payload, self.version = payload, version
        self.means = np.asarray(payload["means"], dtype=float)
        self.scales = np.asarray(payload["scales"], dtype=float)
        self.coef = np.asarray(payload["coefficients"], dtype=float)
        self.categories = payload["categories"]
        self.threshold = float(payload["threshold"])
        self.intercept = float(payload["intercept"])
        self.calibration = np.asarray(payload["calibration"], dtype=float)
        size = len(NUMERIC) + sum(len(c) for c in self.categories)
        if (self.means.shape != (len(NUMERIC),) or self.scales.shape != self.means.shape
                or self.coef.shape != (size,) or len(self.categories) != len(CATEGORICAL)
                or self.calibration.shape != (2,) or not 0 <= self.threshold <= 1
                or np.any(self.scales <= 0)):
            raise ValueError("Invalid model dimensions, scale, or threshold")
        if not all(np.isfinite(x).all() for x in [self.means, self.scales, self.coef,
                                                 self.calibration, self.intercept]):
            raise ValueError("Model contains non-finite parameters")
        if any(len(set(c)) != len(c) or not all(isinstance(v, str) for v in c)
               for c in self.categories):
            raise ValueError("Invalid category vocabulary")

    @classmethod
    def load(cls, path: str | Path):
        raw = Path(path).read_bytes()
        return cls(json.loads(raw), hashlib.sha256(raw).hexdigest()[:16])

    def transform(self, records: list[dict]):
        values = np.asarray([[row[name] for name in NUMERIC] for row in records], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Features must be finite")
        parts = [(values - self.means) / self.scales]
        for name, vocabulary in zip(CATEGORICAL, self.categories):
            parts.append(np.asarray([[float(row[name] == v) for v in vocabulary]
                                     for row in records], dtype=float))
        return np.concatenate(parts, axis=1)

    def predict(self, records: list[dict], calibrated=True):
        score = self.transform(records) @ self.coef + self.intercept
        if calibrated:
            score = self.calibration[0] * score + self.calibration[1]
        return sigmoid(score)

    def unknown_categories(self, records: list[dict]):
        return {name: sum(row[name] not in vocabulary for row in records)
                for name, vocabulary in zip(CATEGORICAL, self.categories)}
