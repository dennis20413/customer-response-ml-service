"""Unlabelled population stability; not a performance or fairness test."""
import argparse
import json
from pathlib import Path

import numpy as np

from .data import load


def reference_for(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Reference input must be nonempty and finite")
    quantiles = np.unique(np.quantile(values, np.linspace(0, 1, 11)))
    if len(quantiles) < 2:
        return {"constant_value": float(quantiles[0]), "proportions": [1.0, 0.0]}
    # Keep finite support bounds and distinct-value midpoints. Two distinct
    # training values must not collapse into a single [-inf, inf] bin.
    cuts = np.concatenate(([np.nextafter(quantiles[0], -np.inf)],
                           quantiles[:-1] + (quantiles[1:] - quantiles[:-1]) / 2,
                           [np.nextafter(quantiles[-1], np.inf)]))
    boundaries = np.concatenate(([-np.inf], cuts, [np.inf]))
    return {"edges": cuts.tolist(),
            "proportions": (np.histogram(values, bins=boundaries)[0] / len(values)).tolist()}


def population_stability(values, spec):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Drift input must be nonempty and finite")
    if "constant_value" in spec:
        same = np.mean(values == spec["constant_value"])
        actual = np.asarray([same, 1 - same])
    else:
        edges = np.asarray(spec["edges"], dtype=float)
        if len(edges) < 2:
            raise ValueError("Reference needs at least two distinct bin edges")
        boundaries = np.concatenate(([-np.inf], edges, [np.inf]))
        actual = np.histogram(values, bins=boundaries)[0] / len(values)
    expected = np.asarray(spec["proportions"], dtype=float)
    # Add epsilon to every bin, then renormalize. Zero-count bins remain represented.
    actual = (actual + 1e-6) / (actual.sum() + 1e-6 * len(actual))
    expected = (expected + 1e-6) / (expected.sum() + 1e-6 * len(expected))
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("artifacts/drift_reference.json"))
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    frame = load(args.csv, require_target=False).iloc[args.start_row:]
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    result = {"rows": len(frame), "diagnostic": "PSI against training-only quantile-midpoint bins with separate out-of-support bins",
              "interpretation": "Descriptive shift indicator; labels are required to measure model degradation",
              "numeric_psi": {name: population_stability(frame[name], spec)
                              for name, spec in reference.items()}}
    serialized = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
