# Model card

## Intended use

Historical research on campaign-response modelling and ML service engineering. Predictions estimate the observed `y` label in UCI Bank Marketing data. They are not recommendations to contact a person, measures of causal marketing lift, investment advice, or credit eligibility scores. No real customer data is accepted by any hosted service in this project; serving is local by default.

## Data

41,188 `bank-additional-full.csv` records, ordered by date (May 2008-November 2010), provided by UCI. Target: subscription to a term deposit. Source: Moro, Rita, and Cortez (2014), DOI `10.24432/C5K306`, CC BY 4.0. This old Portuguese dataset cannot establish performance in Taiwan or on current campaigns.

The 412-row checked-in sample is an offline testing fixture, not the final evaluation sample. It preserves every 100th source row and is not representative.

## Features and exclusions

Inputs: age, days since prior campaign contact (`999` denotes no prior contact), prior contact count, employment variation, consumer price/confidence indices, Euribor, employed count, job, marital status, education, default/housing/loan indicators, prior campaign outcome.

`duration` is excluded because it is only known after the call. Current-call contact channel, campaign count, month, and weekday are excluded. Inputs use a strict shared training/serving schema. Categories unseen in training are mapped to zero one-hot indicators and counted in responses; they do not receive learned category effects.

Age, household and financial descriptors may encode sensitive or protected attributes and proxies. The report's housing/contact slices are descriptive diagnostics, not a fairness audit. It has no consent model, causal estimates, or legal-use assessment. Do not use this prototype for automated individual decisions.

## Training and evaluation

Sequential 50/15/15/20 split: training, score remapping, threshold selection, final test. Scaling, category vocabularies and a regularized logistic model fit only the first partition. One-variable logistic remapping fits the next partition. A demonstration threshold maximizing validation F1 is then frozen. Final-test labels are used only for reporting metrics; tests enforce this boundary.

The raw model's low test ROC AUC and the remapper's **negative slope** expose distribution shift. The learned remapper reverses the order of raw predictions. This is valid to report as a two-stage score model trained on later labels, but it must not be presented as ordinary monotone probability calibration. See the exact slope and intercept in `reports/evaluation.json`. Both the older training prior and fresher calibration-period prior are shown; comparison to the older prior alone would overstate the contribution of modelling.

Raw and remapped results, loss functions, precision/recall, reliability bins and slices are all retained. The numeric threshold selected for remapped scores is used unchanged for raw and baseline F1 reporting, so their F1 values do not represent separately optimized policies. Probability/ranking metrics are more informative for this comparison.

## Limitations and release status

- No customer IDs: clients may recur across partitions. Time order does not prove customer-level isolation.
- No exact timestamps: source row order is the chronology proxy; no calendar gap or seasonal backtest is possible here.
- Label prevalence changes over time. Calibration is not a guarantee against further shift.
- Macro features and category vocabularies may extrapolate poorly across regimes. Unknown values are exposed but no automatic abstention is implemented.
- F1 does not encode contact capacity, consent, outreach costs, fairness or causal uplift. The selected threshold is demonstrative.
- Results were not used to change model features, fit a replacement model, or retune this test set. Future improvements require a new untouched evaluation protocol.
- Research-only release. Production hardening, external validation and independent governance review have not occurred.

## Inference and monitoring

The artifact is human-readable JSON with fixed feature order, scaler parameters, vocabulary, coefficients, remapping parameters, threshold, source hash, partition bounds and library versions. A SHA-256-derived fingerprint identifies exact artifact bytes. The API validates dimensions and finite parameters, and returns 503 if no valid artifact exists. It does not deserialize executable pickle files.

Training-only numeric PSI references flag unlabelled changes. PSI is not a performance, calibration, causal or fairness measure. Evaluate new labels before changing the model or threshold.
