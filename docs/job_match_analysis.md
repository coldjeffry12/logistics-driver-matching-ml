# Job match analysis

This document maps the portfolio prototype to common Machine Learning Engineer
requirements. It is evidence of practical project work, not proof of
production ownership. The project uses synthetic logistics data for learning
and demonstration and does not claim performance on real company data.

| Job Requirement | How This Project Demonstrates It | Evidence in Project | How I Can Explain It in Interview |
|---|---|---|---|
| Python | Uses modular Python for data generation, feature engineering, training, evaluation, inference, APIs, UI, and tests. | `src/`, `tests/`, `scripts/generate_visuals.py` | “I organized the workflow into small modules and reused the same feature logic during training and inference.” |
| SQL | Stores three datasets in SQLite and builds the training join with SQL. | `src/database.py`, `data/logistics.db` after generation | “I used SQL to join shipment, driver, and match tables before feature engineering.” |
| Machine Learning | Implements supervised binary prediction inside a recommendation workflow. | `src/train_model.py`, saved `joblib` artifact | “The model predicts the likelihood that a feasible driver becomes a successful match.” |
| Recommendation Systems | Generates candidates, applies business constraints, scores candidates, and returns top recommendations. | `src/recommender.py` | “It is a pointwise recommendation prototype: score each feasible driver, sort the scores, and return the top five.” |
| Search Ranking | Evaluates order quality within each shipment rather than relying only on classification metrics. | `src/evaluate_model.py`, Precision@5 and NDCG@5 | “I treated each shipment as a query and drivers as ranked results.” |
| Matching Algorithms | Separates hard compatibility rules from softer learned preferences. | `rankable_candidate_mask`, `DriverRecommender.recommend` | “Wrong vehicle, insufficient capacity, and offline status are deterministic filters; ML ranks the remaining candidates.” |
| Data Analysis | Reviews class balance, candidate feasibility, feature importance, and model metrics. | `reports/`, `docs/audit_report.md` | “I used the audit to discover that the first evaluation population did not match inference.” |
| Feature Engineering | Creates explainable distance, utilization, availability, price, risk, response, experience, and history signals. | `src/feature_engineering.py` | “Every model input is intended to be available before dispatch.” |
| Model Training | Trains Logistic Regression and Random Forest with reproducible seeds and pipeline preprocessing. | `src/train_model.py` | “I used Logistic Regression as a baseline and Random Forest as a nonlinear comparison.” |
| Model Evaluation | Uses train, validation, and untouched test shipment groups. | `split_by_shipment`, `reports/model_metrics.json` | “Model selection happens on validation data, and the final numbers come from a separate test set.” |
| Ranking Metrics | Implements Precision@K, NDCG@K, and top-K success by shipment. | `calculate_ranking_metrics` | “Precision@5 measures relevant drivers in the first five; NDCG also rewards correct ordering.” |
| FastAPI | Provides typed health, data, metadata, and recommendation endpoints. | `src/api.py`, `/docs` | “The API is a local model-serving demonstration with request validation and clear error responses.” |
| Streamlit | Provides a recruiter-friendly form, score cards, chart, table, and explanations. | `src/streamlit_app.py` | “The UI lets a nontechnical reviewer enter shipment details and see the ranked output.” |
| Testing | Covers schemas, business constraints, features, ranking metrics, model artifact loading, API behavior, and impossible matches. | `tests/` | “The tests moved beyond smoke checks and now protect the important train/serve assumptions.” |
| Documentation | Includes setup, technical audit, interview answers, demo guidance, resume versions, and GitHub instructions. | `README.md`, `docs/` | “I documented both what the project does and what it does not prove.” |
| Business understanding | Balances feasibility, service quality, risk, distance, and price rather than optimizing a single technical metric. | Data generator, engineered features, hard constraints | “A useful logistics match must be operationally possible before model score matters.” |
| Communication | Provides short and technical explanations for recruiters, interviewers, and GitHub visitors. | `docs/interview_explanation.md`, `docs/demo_script.md` | “I can explain the same system at 15-second, one-minute, and technical-detail levels.” |

## What this project does not prove yet

- It does not prove performance on real logistics or company data.
- It does not prove production deployment, scalability, uptime, or operational
  ownership.
- It does not demonstrate online A/B testing or measurable business impact.
- It does not address marketplace exposure bias, driver fairness, or strategic
  behavior.
- It does not use timestamped point-in-time aggregates or a temporal test split.
- It does not include live traffic, routing APIs, weather, or route restrictions.
- It does not establish probability calibration on real acceptance and
  completion outcomes.
- It does not replace professional experience building production ML systems.

The honest positioning is: this project shows that I can build, audit, test,
explain, and serve an end-to-end ML prototype and that I understand the next
steps required for a real system.
