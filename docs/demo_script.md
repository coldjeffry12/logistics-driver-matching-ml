# Practical demo script

## Opening line

“This is a portfolio prototype that recommends and ranks feasible logistics
drivers for a shipment using synthetic data, explainable business features,
and a scikit-learn model.”

## How to explain the README

Start with the recruiter summary, then point to the workflow image and final
held-out metrics. Do not read the entire README. Mention that setup, API
examples, limitations, and deeper interview material are linked below.

## How to explain the problem statement

“A shipment may have many possible drivers. Some are impossible because their
vehicle or capacity is unsuitable. Among the feasible drivers, the platform
still needs to balance distance, availability, reliability, response time, and
price. This prototype separates those two decisions.”

## How to explain the dataset

“I generated 1,000 shipments, 300 drivers, and 12,000 historical candidate
matches locally. The data contains no real customer or driver information.
The generator includes controlled patterns plus randomness so the task is
learnable but not perfectly deterministic.”

## How to explain feature engineering

Show `src/feature_engineering.py` or the feature-importance image:

“The model receives 12 pre-match inputs. Examples include pickup-distance
score, capacity utilization, rating, availability, price efficiency, delay
and cancellation quality, response speed, historical success, route distance,
and completed-job experience. Vehicle and capacity compatibility are hard
filters rather than model shortcuts.”

## How to explain model training

“I trained Logistic Regression as an interpretable baseline and Random Forest
as a nonlinear comparison. Median imputation is inside each scikit-learn
pipeline, so it is fitted only on training data. Shipments are grouped into
train, validation, and test sets.”

## How to explain evaluation results

Show `docs/screenshots/model_metrics.png`:

“Random Forest was selected using a validation objective that combines 70%
ROC-AUC and 30% NDCG@5. On the untouched test set it achieved ROC-AUC 0.714,
F1 0.547, Precision@5 0.436, and NDCG@5 0.656. These are synthetic-data
results, not expected real-world performance.”

## How to explain corrected metrics

“The first version reported ROC-AUC 0.967. During review I found that the model
was evaluated on many candidates that the live system would filter out for
vehicle or capacity violations. That made the task artificially easy. I
aligned the training and evaluation population with inference, separated
validation from test data, and accepted the lower but more credible result.”

## How to explain FastAPI

Open `http://127.0.0.1:8000/docs`.

“FastAPI provides health, driver, shipment, metadata, and recommendation
endpoints. The request model validates cities, vehicle types, positive values,
and route consistency. The recommendation response has a typed schema.”

## How to explain Streamlit

Open the Streamlit page:

“The demo lets a user enter shipment details and returns the top five feasible
drivers as score cards, a chart, a table, and plain-language reasons. It is a
local demonstration surface, not a production application.”

## How to explain the recommendation result

“The recommender first removes wrong-vehicle, insufficient-capacity, and
offline drivers. It computes the same feature definitions used for training,
loads the saved pipeline, predicts one probability per remaining driver, sorts
the scores, and returns the top five.”

## How to answer why the model is not perfect

“The target includes randomness and unobserved behavior, which reflects that
driver acceptance and delivery success cannot be explained completely by the
available fields. A perfect synthetic score would make me suspicious of
leakage or an overly deterministic label.”

## How to answer why synthetic data

“I did not have permission to use real logistics or personal data. Synthetic
data let me demonstrate the full engineering workflow safely and
reproducibly. I am careful not to treat its metrics as real business evidence.”

## How to answer why not deep learning

“This is structured tabular data with a modest dataset. Logistic Regression
and Random Forest are easier to explain, faster to train, and appropriate
baselines. Deep learning would add complexity without demonstrating a clear
benefit here.”

## What to improve next

“I would use anonymized timestamped data, compute point-in-time history
features, compare learning-to-rank methods, add better routing, calibrate
probabilities, evaluate fairness, and add monitoring and online experiments.”

## Closing line

“The main value of this project is not a high synthetic metric; it is the
audited end-to-end workflow and my ability to explain its design choices,
limitations, and next steps.”
