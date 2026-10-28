"""Official archive download and stable chronological partitions."""
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from .schema import FEATURES, RENAME, Customer

URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
SOURCE = "https://archive.ics.uci.edu/dataset/222/bank+marketing"
EXPECTED_CSV_SHA256 = "74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8"


def download(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=60) as response:
        raw = response.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as outer:
        nested = next(name for name in outer.namelist() if name.endswith("bank-additional.zip"))
        with zipfile.ZipFile(io.BytesIO(outer.read(nested))) as inner:
            member = next(name for name in inner.namelist() if name.endswith("bank-additional-full.csv"))
            content = inner.read(member)
    if hashlib.sha256(content).hexdigest() != EXPECTED_CSV_SHA256:
        raise ValueError("Official CSV content changed; review provenance before retraining")
    path = folder / "bank-additional-full.csv"
    path.write_bytes(content)
    provenance = {"source": SOURCE, "download_url": URL,
                  "archive_sha256": hashlib.sha256(raw).hexdigest(),
                  "csv_sha256": hashlib.sha256(content).hexdigest(),
                  "license": "CC BY 4.0", "creators": "Moro, Rita, Cortez (2014)",
                  "preserved_order": True}
    (folder / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return path


def load(path, require_target=True):
    frame = pd.read_csv(path, sep=";").rename(columns=RENAME)
    expected = FEATURES + (["y"] if require_target else [])
    if len(frame) < (100 if require_target else 1) or not set(expected).issubset(frame.columns):
        raise ValueError("Expected feature schema and nonempty batch (training requires at least 100 rows)")
    if frame[expected].isna().any().any():
        raise ValueError("Missing values violate the input data contract")
    if require_target and not set(frame.y.unique()).issubset({"yes", "no"}):
        raise ValueError("Unexpected target labels")
    # Use precisely the serving contract during training as well.
    for record in frame[FEATURES].to_dict(orient="records"):
        Customer.model_validate(record)
    return frame


def partitions(n):
    if n < 100:
        raise ValueError("At least 100 ordered rows are required")
    boundaries = [0, int(n * .50), int(n * .65), int(n * .80), n]
    return {name: slice(start, stop) for name, start, stop in
            zip(["train", "calibration", "validation", "test"], boundaries, boundaries[1:])}
