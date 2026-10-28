# Measured evaluation

Preserved row order; train 50%, calibration 15%, threshold validation 15%, test 20%

| Test model | Log loss | Brier | AP | ROC AUC | F1 |
|---|---:|---:|---:|---:|---:|
| train_prior_baseline | 0.9774 | 0.2817 | 0.3083 | 0.5000 | 0.0000 |
| calibration_period_prior_baseline | 0.9437 | 0.2785 | 0.3083 | 0.5000 | 0.0000 |
| raw_logistic_regression | 1.7725 | 0.3058 | 0.2271 | 0.3411 | 0.0000 |
| score_remapped_logistic_regression | 0.6111 | 0.2111 | 0.3954 | 0.6589 | 0.4971 |

Frozen validation threshold: `0.103432`. Test rows: `8238`.

Logistic score-remapping slope: `-0.675742`; intercept: `-5.015213`. A negative slope reverses raw ranking. This is not ordinary monotonic calibration.

Calibration and raw scores use the same validation-selected numeric threshold for transparent comparison; the raw model is not separately tuned.

- Old Portuguese campaign data; not evidence of Taiwan generalization
- No customer IDs: recurring customers may cross time partitions
- Row order is chronology proxy; exact timestamps unavailable
- F1 threshold is a demonstration, not a verified business policy
- Calibration can worsen under temporal distribution shift
- Negative remapping slope reverses raw ranking: this is an observed shift failure, not proof of ordinary monotonic calibration

See evaluation.json for source SHA-256, partitions, versions, bins and descriptive slices.
