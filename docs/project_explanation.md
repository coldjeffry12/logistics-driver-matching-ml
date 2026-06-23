# Project explanation

## What this prototype does

This project simulates a logistics platform that receives a shipment order and
ranks drivers or carriers who are likely to complete it successfully. It is a
portfolio prototype built with synthetic data, not a production dispatch
system.

The system first removes drivers whose vehicle type or capacity cannot satisfy
the order, plus drivers marked offline. It then scores the remaining candidates
using a trained scikit-learn model. The final result is a top-five list with a
short, human-readable explanation for each recommendation.

## Data design

The project generates three related datasets:

- `shipments`: route, weight, vehicle requirement, urgency, offered price,
  pickup time, and category.
- `drivers`: location, vehicle and capacity, rating, job history, cancellation
  and delay rates, price, availability, and response time.
- `historical_matches`: candidate pairs and simulated outcomes such as
  acceptance, completion, delay, rating, profit quality, and match success.

The generation rules intentionally create patterns a model can learn. Nearby,
available drivers with good ratings, low risk, reasonable pricing, and suitable
capacity utilization are more likely to become successful matches. Vehicle,
capacity, and offline status are treated as screening rules. Random noise and
unobserved effects prevent the target from being perfectly deterministic.

## Pipeline

1. Generate deterministic synthetic CSV files.
2. Load the three datasets into SQLite.
3. Join them using SQL.
4. Engineer numerical features without fitting preprocessing statistics.
5. Remove candidates that fail the same hard rules used at inference.
6. Split complete shipment IDs into train, validation, and test sets.
7. Fit preprocessing only on training data.
8. Compare Logistic Regression and Random Forest on validation data.
9. Refit the selected model on train plus validation data.
10. Evaluate once on the untouched test set and save with `joblib`.
11. Serve recommendations through FastAPI and Streamlit.

## Features and business rules

The model uses 12 pre-match inputs:

- pickup-distance score
- capacity utilization
- normalized rating
- availability
- offered-price efficiency
- delay and cancellation quality
- response-time quality
- urgency/response compatibility
- historical success rate
- route distance
- completed-job experience

Exact vehicle type, sufficient capacity, and non-offline status are enforced
as deterministic eligibility rules before model scoring. Estimated margin is
retained for reporting only because it overlaps strongly with price
efficiency.

These features are deliberately understandable enough to discuss in an
early-career interview.

## Evaluation

Standard metrics measure binary match prediction: accuracy, precision, recall,
F1, ROC-AUC, and a confusion matrix. Ranking metrics measure recommendation
quality within each shipment: Precision@5, NDCG@5, and top-5 success rate.

The grouped split keeps all candidates for one shipment in one split. Model
selection uses validation data, while the final reported metrics come from an
untouched test set. The current held-out Random Forest results are ROC-AUC
0.714, F1 0.547, Precision@5 0.436, and NDCG@5 0.656.

## Scope and limitations

The labels are synthetic, the location model uses city-level approximate
distance, and the model does not learn from live dispatch feedback. There is
no authentication, monitoring, route traffic, fairness audit, or production
infrastructure. Historical aggregates are static synthetic snapshots rather
than timestamped point-in-time features.

Those limits are intentional: the project demonstrates an end-to-end ML
workflow without claiming production experience.
