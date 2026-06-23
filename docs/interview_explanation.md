# Interview explanation

## 15-second answer

I built a synthetic-data logistics prototype that filters impossible driver
matches and ranks the remaining drivers using explainable features, a
scikit-learn model, FastAPI, and Streamlit.

## 30-second answer

I built an end-to-end driver-matching portfolio prototype for shipment orders.
It generates synthetic data, stores it in CSV and SQLite, engineers pre-match
distance, price, availability, quality, and risk features, and compares
Logistic Regression with Random Forest. The system applies vehicle, capacity,
and offline-status rules before returning the top five drivers through
FastAPI and Streamlit.

## 60-second answer

I built a portfolio prototype that recommends drivers for a logistics
shipment. Because I did not have access to real company data, I generated
synthetic shipment, driver, and historical match records locally and joined
them through SQLite and SQL. I engineered explainable pre-match features for
distance, capacity utilization, rating, availability, price efficiency,
response time, reliability, and experience. Exact vehicle type, sufficient
capacity, and non-offline status are enforced as hard business rules before
the model scores drivers. I compared Logistic Regression and Random Forest
using shipment-grouped train, validation, and untouched test sets. Random
Forest was selected using a combined ROC-AUC and NDCG@5 validation objective.
The final model is served through FastAPI and a Streamlit demo. It is a
portfolio prototype, not a production dispatch system.

## 2-minute technical answer

The project models driver recommendation as two stages. The first is
deterministic eligibility screening: a candidate must have the exact required
vehicle, enough payload capacity, and must not be offline. The second stage is
ML ranking among feasible candidates.

The generator creates 1,000 shipments, 300 drivers, and 12,000 historical
candidate matches. Those tables are stored in CSV and SQLite and joined with
SQL. Feature engineering creates 15 explainable signals; 12 are used by the
model, two are hard compatibility rules, and estimated margin is retained for
reporting only. Missing values remain missing until the scikit-learn pipeline,
so median imputation is learned from training data only.

Candidates are split by shipment into training, validation, and test groups.
I train Logistic Regression as an interpretable baseline and Random Forest as
a nonlinear alternative. Model selection uses a fixed validation objective:
70% ROC-AUC and 30% NDCG@5. The selected Random Forest is refitted on training
plus validation data and evaluated once on 150 untouched shipments. It reached
ROC-AUC 0.714, F1 0.547, Precision@5 0.436, and NDCG@5 0.656.

The saved `joblib` pipeline is loaded by the recommender. For a new shipment,
the system computes the same features, filters infeasible drivers, predicts
scores, sorts them, and returns the top five with rule-based explanations.
FastAPI exposes the workflow as an API, while Streamlit provides a public
portfolio demo.

## Simple recruiter explanation

The project demonstrates that I can take a business problem from data design
through model training, testing, and a working demonstration. It uses only
synthetic data and does not claim production deployment. It is particularly
relevant because it combines my existing software support, SQL, testing, and
troubleshooting skills with practical machine learning work.

## Technical ML interviewer explanation

The core design choice is separating candidate eligibility from learned
ranking. Hard constraints are removed before training and inference, so the
evaluation population matches the production-style scoring population.
Preprocessing is encapsulated in model pipelines, splitting is grouped by
shipment, model selection happens on validation data, and final metrics come
from an untouched test set. Ranking metrics are calculated per shipment.

## Problem being solved

A logistics platform can have many possible drivers for one shipment.
Reviewing them manually is slow, and a pure rule-based approach may not rank
quality well. The prototype first guarantees operational feasibility and then
ranks feasible drivers using distance, availability, reliability, response
time, price, and experience.

## Dataset explanation

The dataset is generated locally and contains:

- 1,000 shipment orders with routes, weights, urgency, offered price, pickup
  hour, and category.
- 300 drivers with city, vehicle, capacity, rating, completed jobs, risk
  rates, price, availability, response time, and historical success.
- 12,000 historical candidate matches with simulated acceptance, completion,
  delay, customer rating, and target outcomes.

No real personal or company data is used.

## Target variable explanation

`match_success` equals one when the candidate is eligible, accepts the
shipment, completes it, has no more than 60 minutes of delay, and receives a
customer rating of at least 3.5. It equals zero otherwise. The generator adds
random effects so the label is not perfectly recoverable.

## Feature engineering explanation

The model inputs are pickup-distance score, capacity utilization, normalized
rating, availability, price efficiency, delay quality, cancellation quality,
response-time score, urgency match, historical success, log route distance,
and log completed jobs. Vehicle match and capacity match are hard rules.
Estimated margin is reporting-only because it strongly overlaps with price
efficiency.

## Model training explanation

Shipments are divided into 70% training, 15% validation, and 15% test groups.
All candidates for one shipment remain together. Each model uses a
scikit-learn pipeline with median imputation. Logistic Regression also uses
standard scaling. Models are compared on validation data, and the winner is
refitted on training plus validation data before final testing.

## Why Logistic Regression was used

Logistic Regression is a strong, interpretable baseline for binary
classification. It provides a useful reference for whether nonlinear
interactions justify a more flexible model. It also makes preprocessing,
class weighting, and coefficient-based interpretation straightforward.

## Why Random Forest was selected

Random Forest was not better on every metric. Logistic Regression had slightly
higher validation ROC-AUC, recall, and F1. Random Forest had better
Precision@5 and NDCG@5, and it narrowly won the predefined combined objective
of 70% ROC-AUC and 30% NDCG@5. That combined objective reflects both
classification discrimination and ranking order.

## Ranking explanation

The model predicts a score for each feasible driver. Drivers are sorted from
highest to lowest score, and the first five are returned. This is pointwise
ranking: a binary classifier produces candidate scores, and sorting converts
them into a recommendation list.

## ROC-AUC explanation

ROC-AUC measures how often the model ranks a random successful match above a
random unsuccessful match across all possible classification thresholds. A
score of 0.5 is random ordering and 1.0 is perfect discrimination.

## F1-score explanation

F1 is the harmonic mean of precision and recall at the selected 0.5 threshold.
It balances the cost of recommending unsuccessful drivers against missing
potentially successful drivers.

## Precision@5 explanation

Precision@5 is the proportion of the first five ranked drivers that are
successful historical matches, averaged across shipments. The final value is
0.436.

## NDCG@5 explanation

NDCG@5 measures ranking quality and gives more credit when successful drivers
appear nearer the top. It normalizes against the best possible ordering for
each shipment. The final value is 0.656.

## Top-5 success explanation

Top-5 success is the percentage of shipments where at least one successful
candidate appears in the first five. The value is 0.947, but it is less
discriminating because many synthetic shipments have several positive
candidates.

## Data leakage explanation

Post-outcome fields such as acceptance, completion, delivery delay, customer
rating, final profit outcome, and `match_success` are not model inputs.
Preprocessing is fitted inside the model pipeline. All candidates for a
shipment remain in one split. For real data, historical driver aggregates
would also need point-in-time calculation to ensure they do not include future
jobs.

## What was wrong in the first version and how it was fixed

The first version reported ROC-AUC 0.967. It evaluated many drivers who failed
vehicle or capacity rules even though live inference filtered them, making
negative cases artificially easy. The generator also allowed 202 impossible
successful labels, imputation occurred before splitting, and model comparison
used the test set.

I fixed this by enforcing hard eligibility in the target, training and
evaluating only rankable candidates, moving imputation into pipelines, adding
separate validation and test sets, removing duplicate availability
adjustment, and expanding tests. The corrected held-out ROC-AUC is 0.714.

## Main limitations

- All data and labels are synthetic.
- Driver history is a static snapshot rather than a timestamped aggregate.
- City-level distances are approximate.
- The model score is not calibrated on real dispatch outcomes.
- Candidate exposure and marketplace selection bias are not modeled.
- There is no live traffic, fairness audit, monitoring, authentication, or
  online experiment.

## Future improvements

- Train on anonymized timestamped logistics outcomes.
- Build point-in-time historical features.
- Compare pairwise and listwise learning-to-rank models.
- Add route-aware traffic and service-area constraints.
- Calibrate probabilities and optimize business-specific costs.
- Add model monitoring, drift checks, fairness analysis, and CI/CD.

## What I personally learned

I learned that candidate definition and evaluation design can matter more than
the choice between two models. I also learned to separate hard constraints
from learned preferences, keep preprocessing inside pipelines, preserve a
true test set, test business assumptions, and communicate corrected metrics
honestly.

## What I personally built

I built the scoped prototype workflow: synthetic data generation, CSV and
SQLite storage, SQL joins, feature engineering, model comparison, grouped
evaluation, recommendation logic, FastAPI endpoints, Streamlit interface,
tests, and documentation. I used AI-assisted coding as a development tool, but
I reviewed the design, ran the project, investigated the inflated first
result, corrected the evaluation, and prepared the explanations and
limitations. I describe this as portfolio work, not professional production
ML ownership.

## What `driver_id` means

`driver_id` is a synthetic identifier such as `DRV-0111`. It links a generated
driver record to candidate matches and recommendations. It is not a real
person, employee number, account, or customer identifier.

## My current gaps

My main gaps are professional production ML experience, real timestamped
logistics data, cloud model operations, monitoring, fairness evaluation, and
online experimentation. I can explain the prototype and its engineering
choices, but I would seek guidance for production architecture and
business-critical model decisions.

## How this connects to a Machine Learning Engineer job

The project demonstrates Python, SQL, data pipelines, feature engineering,
model training, grouped evaluation, ranking metrics, model persistence,
FastAPI serving, Streamlit demonstration, testing, debugging, and technical
communication. It does not replace production ML experience, but it provides
concrete evidence that I understand the workflow and can build a scoped,
reviewable prototype.
