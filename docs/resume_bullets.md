# Resume and portfolio wording

## Short resume project title

**Logistics Driver Matching ML Prototype**

## Version A: 3 short resume bullets

- Built a synthetic-data logistics driver-matching prototype using Python,
  scikit-learn, SQL/SQLite, FastAPI, and Streamlit.
- Engineered pre-match distance, availability, pricing, capacity, reliability,
  and response-time signals and ranked the top five feasible drivers.
- Compared Logistic Regression and Random Forest using grouped validation,
  held-out testing, ROC-AUC, F1, Precision@5, and NDCG@5.

## Version B: 5 standard resume bullets

- Built an end-to-end logistics driver-matching portfolio prototype using
  locally generated synthetic data and Python.
- Developed a reproducible pipeline that stores shipment, driver, and match
  records in SQLite and creates the training dataset through SQL joins.
- Engineered explainable matching features while enforcing vehicle type,
  payload capacity, and offline status as deterministic business constraints.
- Trained and compared Logistic Regression and Random Forest with
  shipment-grouped train, validation, and held-out test splits.
- Implemented and tested a top-five recommendation engine, FastAPI service,
  and Streamlit demo with classification and ranking evaluation.

## Version C: 7 detailed LinkedIn or portfolio bullets

- Designed a logistics recommendation prototype that scores and ranks feasible
  drivers for shipment orders without using real customer or company data.
- Generated 1,000 synthetic shipments, 300 drivers, and 12,000 historical
  candidate matches with controlled logistics relationships and random effects.
- Stored source datasets in CSV and SQLite and implemented SQL joins for
  shipment, driver, and historical match data.
- Engineered 15 explainable signals covering pickup distance, capacity
  utilization, availability, price efficiency, driver quality, delivery risk,
  response speed, route length, and experience.
- Compared Logistic Regression and Random Forest using scikit-learn pipelines,
  class weighting, shipment-grouped validation, and an untouched test set.
- Evaluated the selected Random Forest with ROC-AUC 0.714, F1 0.547,
  Precision@5 0.436, and NDCG@5 0.656 on synthetic held-out data.
- Built pytest coverage, typed FastAPI endpoints, a Streamlit demonstration,
  recruiter documentation, and reproducible PowerShell run scripts.

## LinkedIn project description

Built an end-to-end logistics driver-matching portfolio prototype using
Python, pandas, scikit-learn, SQLite, FastAPI, Streamlit, and pytest. The
project generates synthetic shipment, driver, and historical match data,
creates a SQL-based training dataset, and engineers explainable signals for
pickup distance, capacity utilization, availability, pricing, driver quality,
response time, and delivery risk. Vehicle type, payload capacity, and offline
status are enforced as hard business constraints before model scoring. I
compared Logistic Regression and Random Forest using shipment-grouped train,
validation, and held-out test splits, then evaluated classification and
ranking performance with ROC-AUC, F1, Precision@5, NDCG@5, and confusion
matrices. I also created a top-five recommendation engine, typed API, local
demo, automated tests, visual reports, and interview documentation. This is a
synthetic-data portfolio prototype, not a production deployment or evidence
of real operational impact.

## GitHub repository description

Synthetic-data ML prototype for ranking feasible logistics drivers with scikit-learn, FastAPI, Streamlit, SQLite, and ranking metrics.
