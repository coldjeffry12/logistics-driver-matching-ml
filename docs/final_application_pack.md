# Final Application Pack

All wording below describes this repository as an early-career portfolio prototype built with synthetic data. It does not claim production deployment, real company data, operational adoption, or measured business impact.

## 1. Resume Project Section

### Logistics Driver Matching and Recommendation System

*Machine Learning Portfolio Prototype | Python, SQL, scikit-learn, FastAPI, Streamlit*

- Built a reproducible logistics matching prototype using synthetic shipment, driver, and candidate-history data, with Python pipelines and SQL/SQLite storage.
- Engineered explainable feasibility, distance, capacity, reliability, response-time, pricing, and experience features while excluding post-outcome leakage fields.
- Compared Logistic Regression and Random Forest models using shipment-grouped train, validation, and held-out test splits; evaluated classification and ranking performance with ROC-AUC, F1, Precision@5, NDCG@5, and top-five success rate.
- Developed a top-five driver recommendation engine that applies vehicle, capacity, and availability rules before ranking feasible candidates with the selected Random Forest pipeline.
- Exposed the prototype through FastAPI and Streamlit, saved the trained scikit-learn pipeline with joblib, and added eight pytest tests covering data, features, ranking, recommendations, model loading, and API integration.

## 2. JobStreet Profile Summary

Early-career Computer Science graduate with experience in Java application support, SQL and database validation, XML/EAI validation, system testing, deployment preparation, troubleshooting, and technical documentation. I am developing toward Machine Learning Engineer roles through practical portfolio work, including a synthetic-data logistics driver matching prototype built with Python, SQLite, scikit-learn, FastAPI, Streamlit, and pytest. My background supports careful data validation, reproducible testing, issue investigation, and clear documentation, while my current projects demonstrate growing capability in feature engineering, model evaluation, recommendation ranking, and ML application development.

## 3. Cover Letter Paragraph

To strengthen my transition into machine learning engineering, I built a logistics driver matching and recommendation portfolio prototype using synthetic data. The project combines Python and SQL/SQLite data pipelines, explainable feature engineering, Logistic Regression and Random Forest modeling, shipment-grouped evaluation, and top-five ranking metrics. I also implemented a recommendation service with FastAPI, an interactive Streamlit demo, and pytest coverage. Although it is not a production system, the project demonstrates how I approach logistics feasibility rules, candidate ranking, model evaluation, testing, and technical communication in a structured and honest way.

## 4. Recruiter Message

Hello, I am an early-career Computer Science graduate applying for Machine Learning Engineer opportunities. My background includes Java application support, SQL/database validation, system testing, troubleshooting, and documentation. I recently completed a synthetic-data logistics driver matching portfolio prototype using Python, scikit-learn, SQLite, FastAPI, Streamlit, and pytest. I would appreciate your consideration for roles where I can contribute these foundations while continuing to grow in applied machine learning.

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
