# Final Project Pitch

## 1. One-sentence project pitch

I built a synthetic-data logistics matching prototype that filters infeasible
drivers and ranks the top five candidates using explainable features,
scikit-learn, FastAPI, and Streamlit.

## 2. 30-second recruiter explanation

This project demonstrates an end-to-end early-career ML engineering workflow.
It generates synthetic shipment and driver data, joins it through SQLite and
SQL, engineers explainable pre-match features, compares Logistic Regression
with Random Forest, and returns the top five feasible drivers. It also includes
grouped evaluation, FastAPI, a public Streamlit demo, pytest, and GitHub
Actions. It is a portfolio prototype, not a production dispatch system.

## 3. 60-second interview explanation

I built a logistics driver matching and recommendation prototype using
synthetic data because I did not have permission to use real logistics or
personal data. The system first applies hard rules for exact vehicle type,
payload capacity, and non-offline status. It then ranks feasible drivers using
pre-match signals such as pickup distance, capacity utilization, rating,
availability, price efficiency, response time, delivery-risk history, and
experience. I compared Logistic Regression with Random Forest using
shipment-grouped train, validation, and held-out test splits. Random Forest
was selected using a predefined combination of ROC-AUC and NDCG@5. The final
held-out results include ROC-AUC 0.714, F1 0.547, Precision@5 0.436, and
NDCG@5 0.656. I exposed the workflow through FastAPI and a public Streamlit
demo and added 10 tests. These results describe synthetic data only.

## 4. Technical explanation

The project separates candidate eligibility from ML ranking. CSV and SQLite
tables contain generated shipments, drivers, and candidate outcomes. SQL joins
produce the candidate dataset, and shared feature engineering supports both
training and inference. Post-outcome fields are excluded from model inputs,
median imputation is fitted inside scikit-learn pipelines, and all candidates
for one shipment remain in the same split. Logistic Regression provides an
interpretable baseline; Random Forest captures nonlinear tabular interactions.
The recommender filters infeasible drivers, predicts one score per remaining
candidate, sorts descending, and returns five synthetic driver IDs with
rule-based explanations.

## 5. Simple explanation for non-technical people

The prototype takes a delivery request, removes drivers who cannot do the job,
and then orders the remaining drivers using factors such as distance,
availability, rating, reliability, response time, and price. It shows the five
best candidates and explains the recommendation. Every shipment and driver in
the demo is invented for learning and demonstration.

## 6. What the project contributes

- A reproducible Python and SQL/SQLite data workflow.
- A clear separation between logistics rules and learned ranking.
- Honest grouped classification and ranking evaluation.
- A working API, public demo, automated tests, CI, and technical documentation.
- Evidence of practical early-career ML engineering skills and careful
  technical communication.

## 7. What the project does not prove

- It does not prove performance on real logistics data.
- It does not prove production deployment, scale, uptime, or business impact.
- It does not use real customer, driver, shipment, or company information.
- It does not include production monitoring, security, fairness validation,
  real routing, or online experimentation.

## 8. Best resume bullet version

- Built and tested a synthetic-data logistics driver-matching prototype using
  Python, SQL/SQLite, scikit-learn, FastAPI, and Streamlit; engineered
  explainable pre-match features, compared Logistic Regression and Random
  Forest with shipment-grouped evaluation, and returned top-five feasible
  recommendations measured with ROC-AUC, F1, Precision@5, and NDCG@5.

## 9. Best LinkedIn post version

I built a logistics driver matching and recommendation portfolio prototype
using synthetic data. The project separates hard vehicle, capacity, and
availability rules from ML ranking, then compares Logistic Regression and
Random Forest with shipment-grouped evaluation. It includes Python,
SQL/SQLite, explainable feature engineering, FastAPI, a public Streamlit demo,
10 pytest tests, and GitHub Actions.

The corrected held-out Random Forest results are ROC-AUC 0.714, F1 0.547,
Precision@5 0.436, and NDCG@5 0.656. These results describe synthetic data
only; the project is not presented as a production system or evidence of
business impact.

GitHub: https://github.com/coldjeffry12/logistics-driver-matching-ml  
Live demo: https://logistics-driver-matching-ml-aldmp73kjkhhj9jbuhl4hx.streamlit.app

## 10. Best recruiter message version

Hi, I would like to share my Machine Learning portfolio project: a logistics
driver matching and recommendation prototype using synthetic data. It
demonstrates Python, SQL/SQLite, scikit-learn, feature engineering, grouped
classification and ranking evaluation, FastAPI, Streamlit, pytest, and GitHub
Actions. My background is in Java application support, SQL validation,
testing, troubleshooting, and documentation, and I am developing these
foundations toward an early-career Machine Learning Engineer role.

GitHub: https://github.com/coldjeffry12/logistics-driver-matching-ml  
Live demo: https://logistics-driver-matching-ml-aldmp73kjkhhj9jbuhl4hx.streamlit.app
