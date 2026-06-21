# Final Application Pack

All wording below describes this repository as an early-career portfolio prototype built with synthetic data. It does not claim production deployment, real company data, operational adoption, or measured business impact.

## Public Project Links

- GitHub: https://github.com/coldjeffry12/logistics-driver-matching-ml
- Live demo: https://logistics-driver-matching-ml-aldmp73kjkhhj9jbuhl4hx.streamlit.app

## 1. Resume Project Section

### Logistics Driver Matching and Recommendation System

*Machine Learning Portfolio Prototype | Python, SQL, scikit-learn, FastAPI, Streamlit*

- Built a reproducible logistics matching prototype using synthetic shipment, driver, and candidate-history data, with Python pipelines and SQL/SQLite storage.
- Engineered explainable feasibility, distance, capacity, reliability, response-time, pricing, and experience features while excluding post-outcome leakage fields.
- Compared Logistic Regression and Random Forest models using shipment-grouped train, validation, and held-out test splits; evaluated classification and ranking performance with ROC-AUC, F1, Precision@5, NDCG@5, and top-five success rate.
- Developed a top-five driver recommendation engine that applies vehicle, capacity, and availability rules before ranking feasible candidates with the selected Random Forest pipeline.
- Exposed the prototype through FastAPI and Streamlit, saved the trained scikit-learn pipeline with joblib, and added ten pytest tests covering data, features, ranking, recommendations, model loading, API integration, and deployment-time artifact readiness.

## 2. JobStreet Profile Summary

Early-career Computer Science graduate with experience in Java application support, SQL and database validation, XML/EAI validation, system testing, deployment preparation, troubleshooting, and technical documentation. I am developing toward Machine Learning Engineer roles through practical portfolio work, including a synthetic-data logistics driver matching prototype built with Python, SQLite, scikit-learn, FastAPI, Streamlit, and pytest. My background supports careful data validation, reproducible testing, issue investigation, and clear documentation, while my current projects demonstrate growing capability in feature engineering, model evaluation, recommendation ranking, and ML application development.

## 3. Cover Letter Paragraph

To strengthen my transition into machine learning engineering, I built a logistics driver matching and recommendation portfolio prototype using synthetic data. The project combines Python and SQL/SQLite data pipelines, explainable feature engineering, Logistic Regression and Random Forest modeling, shipment-grouped evaluation, and top-five ranking metrics. I also implemented a recommendation service with FastAPI, an interactive Streamlit demo, and pytest coverage. Although it is not a production system, the project demonstrates how I approach logistics feasibility rules, candidate ranking, model evaluation, testing, and technical communication in a structured and honest way.

## 4. Recruiter Message

Hi, I would like to share my Machine Learning portfolio project: a logistics driver matching and recommendation prototype using synthetic data. It demonstrates Python, SQL/SQLite, scikit-learn, feature engineering, ranking evaluation, FastAPI, Streamlit, and pytest testing.

GitHub: https://github.com/coldjeffry12/logistics-driver-matching-ml  
Live demo: https://logistics-driver-matching-ml-aldmp73kjkhhj9jbuhl4hx.streamlit.app

I am an early-career Computer Science graduate with experience in Java application support, SQL/database validation, system testing, troubleshooting, and documentation. I would appreciate consideration for roles where I can apply these foundations while continuing to grow in machine learning engineering.

## 5. Interview 60-Second Explanation

I built a logistics driver matching and recommendation prototype using synthetic data to demonstrate an end-to-end machine learning workflow. The system first applies hard feasibility rules for vehicle type, payload capacity, and driver availability. It then engineers signals such as pickup distance, capacity utilization, rating, historical delay and cancellation rates, response time, price efficiency, and experience. I compared Logistic Regression with Random Forest using shipment-grouped train, validation, and held-out test splits so candidates from one shipment could not leak across splits. Random Forest was selected using classification and ranking performance, with held-out ROC-AUC of 0.714, F1 of 0.547, Precision@5 of 0.436, and NDCG@5 of 0.656. I packaged the result as a top-five recommender, FastAPI service, Streamlit demo, and pytest suite. It is a learning and demonstration project, not a production system, and it uses no real company or personal data.

## 6. Interview 30-Second Explanation

I built a synthetic-data logistics matching prototype that filters infeasible drivers using vehicle, capacity, and availability rules, then ranks the remaining candidates with a scikit-learn model. I compared Logistic Regression and Random Forest with shipment-grouped evaluation and measured both classification and top-five ranking quality. I also added a FastAPI service, Streamlit demo, SQLite pipeline, and pytest tests. It is an early-career portfolio project rather than a production deployment.

## 7. Honest Gap Explanation

I do not have 13 years of professional machine learning experience, and I would not present myself as equivalent to a senior ML engineer. What I can offer is a relevant Computer Science foundation, practical experience in application support, SQL validation, testing, troubleshooting, and documentation, plus evidence that I am building applied ML skills through complete and reviewable projects. I am suitable for an early-career role where disciplined engineering habits, learning ability, and honest technical reasoning are valued, with appropriate guidance for production-level decisions.

## 8. Synthetic Data Explanation

I used synthetic data because I did not have access to suitable anonymized logistics dispatch data and did not want to imply access to confidential company, customer, shipment, or driver information. Synthetic data let me demonstrate the pipeline, feature engineering, grouped evaluation, and recommendation design while clearly acknowledging that the results do not establish real-world performance.

## 9. Production Readiness Explanation

No. This is a portfolio prototype for learning and demonstration. A production version would require real point-in-time data, stronger data validation, routing and operational constraints, security and authentication, monitoring, drift detection, retraining controls, fairness review, load testing, deployment infrastructure, and validation with business stakeholders.

## 10. GitHub Description

Synthetic-data ML portfolio prototype for logistics driver matching, top-five ranking, FastAPI serving, Streamlit demo, and pytest testing.

## 11. GitHub Topics

- machine-learning
- recommendation-system
- ranking
- logistics
- driver-matching
- synthetic-data
- python
- scikit-learn
- sql
- sqlite
- feature-engineering
- fastapi
- streamlit
- pytest
- ml-portfolio

## 12. LinkedIn Project Description

I built a logistics driver matching and recommendation portfolio prototype
using synthetic data. The project combines Python and SQL/SQLite data
pipelines, explainable feature engineering, Logistic Regression and Random
Forest comparison, shipment-grouped evaluation, and top-five ranking metrics.
It includes a FastAPI service, public Streamlit demonstration, pytest coverage,
and GitHub Actions. The project is intended to demonstrate early-career ML
engineering skills and is not presented as a production system or evidence of
real logistics business impact.

GitHub: https://github.com/coldjeffry12/logistics-driver-matching-ml  
Live demo: https://logistics-driver-matching-ml-aldmp73kjkhhj9jbuhl4hx.streamlit.app

## 13. One-line Project Pitch

An end-to-end synthetic-data ML portfolio prototype that filters infeasible
drivers and ranks the top five candidates for logistics shipments.

## 14. Three Strongest Technical Selling Points

1. Separates deterministic logistics feasibility rules from learned candidate ranking.
2. Uses shipment-grouped train, validation, and held-out test splits with both classification and ranking metrics.
3. Demonstrates a reproducible path from Python/SQLite data generation through model training, FastAPI, Streamlit, pytest, and GitHub Actions.

## 15. Three Limitations to Explain Honestly

1. The dataset and labels are synthetic, so the metrics do not establish performance on real logistics operations.
2. The public Streamlit app is a portfolio demonstration with first-use training, not a production serving architecture.
3. The project does not include real routing, timestamped point-in-time features, monitoring, fairness validation, security controls, or online experimentation.
