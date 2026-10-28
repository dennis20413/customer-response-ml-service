# Validation record

Executed on 2026-10-28 with Python 3.11.6 on Windows. Dependency versions are pinned in `requirements.txt` and `requirements-dev.txt`; no dependency downgrade was used.

| Check | Result |
|---|---|
| `python -m pytest -q` | 25 passed in 2.74 seconds; one Starlette notice that the supported httpx adapter is deprecated |
| `python -m response_service.train --csv data/bank-additional-full.csv` | Completed on all 41,188 official rows; 8,238 final-test rows; safe JSON scoring matched sklearn output |
| Final-test-label perturbation regression | Passing; coefficients, preprocessing, score remapping and threshold remain identical |
| API integration via TestClient | Passing for readiness, metadata, valid prediction, unknown categories, invalid inputs, batch limits and unavailable artifacts |
| Real local HTTP smoke through uvicorn | `/health`, `/model`, `/predict` returned 200; extra `duration` and empty batch returned 422; prediction probability valid and model fingerprint matched |
| `python -m response_service.drift --csv data/bank-additional-full.csv --start-row 32950 --output reports/drift-diagnostic.json` | Completed on 8,238 final-period rows; two-value and constant-feature regressions pass; shifted macroeconomic inputs flag strong out-of-support changes |
| Docker build/run | Recipe supplied; not executed in this environment |
| GitHub Actions | Workflow supplied; local validation does not establish a remote CI result |

The Windows environment used for file-only work restricted local networking and blocked Python's asyncio socketpair initialization. The successful API tests ran with local networking enabled. This was an execution-environment restriction; the same unchanged suite passed after that restriction was removed.

The full-data evaluation is regenerated from the fixed model/protocol. Additional reporting adds the calibration-period prior, partition prevalence and explicit ranking-reversal disclosure; it does not retune the model after observing the test set. The SHA-256-derived model fingerprint remained `fdeb6c03d678f7aa`.

Measured report and provenance: `reports/evaluation.json`, `reports/evaluation.md`, `reports/data-provenance.json`. They describe a historical stress test, not an external validation or production deployment.
