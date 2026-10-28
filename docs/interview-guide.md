# Interview walkthrough

## Five-minute demo

1. Run the tests. Explain the test that changes only final-test labels and proves the model and threshold do not change.
2. Open `reports/evaluation.md`. Compare raw logistic regression with both class-prior baselines, then explain why fresher labels matter.
3. Show the negative remapping slope. Explain that it reverses ranking and signals a temporal failure of the raw model; avoid calling it guaranteed probability calibration.
4. Start the API and submit `examples/request.json`. Add `duration` and show the validation error, then send an unfamiliar job and show the unknown-category count.
5. Compare a prediction to the saved JSON parameters and show the model fingerprint. Explain the numerical equivalence check against sklearn.

## Questions to prepare for

**Why not randomly split the rows?** The official file is date-ordered. A random split shares historical regimes with future examples and understates the difficulty of future-period evaluation.

**What is wrong with `duration`?** It is measured after a call and cannot be known for a pre-call estimate. Using it answers a different question and inflates the apparent performance of the intended task.

**Why keep a weak raw model?** This project makes a failure reproducible rather than selecting a new model after observing the final test. A new research iteration should define a fresh validation/holdout plan. The engineering value is not tied to a flattering metric.

**Does the negative calibration slope improve calibration?** It is an unconstrained score remapping, so a negative coefficient reverses ordering. Describe what it actually does. It is not monotone calibration, and observed test improvement is not a promise about new populations.

**Why include the calibration-period prior?** The remapper sees newer labels than the original model. A prior estimated from those same labels is a fairer baseline for the effect of recent prevalence information.

**How do you avoid training/serving skew?** Export feature order, means/scales, one-hot vocabularies and all score parameters together. Verify JSON inference against the fitted pipeline to tight tolerance before producing the report. Use the same input contract during data loading and API requests.

**Would you put this in production?** No. It needs current consented data, temporal and customer-isolated external validation, an appropriate business utility policy, fairness assessment, auth/TLS, monitoring, model approval and rollback. These are future requirements, not implemented accomplishments.

## Honest resume framing

After you reproduce and understand the work, describe the software and methodology: built a Python ML pipeline with ordered train/calibration/validation/test partitions, leakage controls, baseline comparisons, safe model export, validated prediction API and automated regression checks. If you quote performance, identify the historical dataset, test sample, protocol and ranking-reversal caveat. Do not claim marketing lift, employer usage, live traffic or a production deployment.
