# Technical audit report

## Overall assessment

Score after remediation: **86/100**.

The project is suitable for an early-career Machine Learning Engineer portfolio
when presented explicitly as a synthetic-data prototype. Its strongest aspect
is now the alignment between business rules, model training, evaluation, and
inference. It should not be presented as evidence of production deployment or
real-world model performance.

## Material findings

### Fixed

1. The original evaluation included many wrong-vehicle and insufficient-
   capacity rows even though inference filtered them. This inflated ROC-AUC.
2. The generator allowed 202 impossible candidates to receive successful
   labels. Successful labels now require all hard constraints.
3. Feature engineering filled missing values before the grouped split. Missing
   values now flow into scikit-learn pipelines, whose imputers are fitted on
   training data only.
4. The original test set was used to compare models. The pipeline now uses
   shipment-grouped train, validation, and untouched test sets.
5. Availability was represented in the model and multiplied into the score a
   second time. The manual multiplier was removed; offline drivers are filtered
   and busy/available status remains a model feature.
6. The original tests were mainly schema checks. The suite now covers hard-
   constraint label validity, missing-value behavior, metric definitions,
   impossible recommendations, artifact metadata, and API requests.

### No direct target leakage found

The model feature list does not contain `accepted`,
`completed_successfully`, `delivery_delay_minutes`, `customer_rating`,
`final_profit_score`, or `match_success`. These post-outcome fields remain in
historical data only.

## Feature review

| Signal | Pre-match? | Decision |
|---|---|---|
| Distance score | Yes | Keep as model input |
| Vehicle type match | Yes | Keep as hard rule, remove from model |
| Capacity match | Yes | Keep as hard rule, remove from model |
| Capacity utilization | Yes | Keep as model input |
| Rating score | Yes, if snapshot is current | Keep |
| Availability flag | Yes | Keep; offline drivers are screened out |
| Price efficiency | Yes | Keep |
| Delay risk score | Yes, if point-in-time | Keep |
| Cancellation risk score | Yes, if point-in-time | Keep |
| Response-time score | Yes, if point-in-time | Keep |
| Urgency match score | Yes | Keep |
| Historical success rate | Yes, if point-in-time | Keep with leakage warning |
| Estimated margin score | Yes | Reporting only; redundant with price ratio |
| Route distance log | Yes | Keep |
| Completed jobs log | Yes, if point-in-time | Keep |

For real data, every historical aggregate must be computed as of the
recommendation timestamp. Computing it using future jobs would introduce
leakage.

## Evaluation interpretation

The selected Random Forest achieved held-out ROC-AUC 0.714 and NDCG@5 0.656.
These are moderate and plausible for the synthetic task. Top-5 success is
0.947, but that metric is less discriminating because many synthetic shipments
have multiple positive candidates. Precision@5 and NDCG@5 should receive more
attention in interviews.

The metrics show that the model learns the generator's programmed patterns.
They do not estimate performance on a real logistics marketplace.

## Remaining risks

- Synthetic labels are based on assumptions rather than observed behavior.
- Driver aggregates are static snapshots.
- Candidate generation does not model marketplace exposure or selection bias.
- Predicted probabilities are not calibrated against real outcomes.
- There is no temporal split, traffic data, fairness analysis, monitoring, or
  online experiment.
- The UI and API are local demonstration surfaces.

## Resume decision

Safe to include with language such as "portfolio prototype," "synthetic data,"
and "held-out evaluation." Do not claim production deployment, real company
data, revenue impact, or operational adoption.
