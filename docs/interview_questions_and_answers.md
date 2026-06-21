# Project interview questions and answers

These answers are intentionally modest. They describe a portfolio prototype,
not professional production ML ownership.

## 1. What problem does the project solve?

It ranks feasible drivers for a shipment. Business rules first remove drivers
with the wrong vehicle, insufficient capacity, or offline status. A model then
orders the remaining drivers by predicted match success.

## 2. Why did you choose logistics matching?

It connects recommendation and ranking methods with understandable operational
constraints. It is also relevant to companies that operate marketplaces,
dispatch systems, transportation platforms, or delivery networks.

## 3. Is the data real?

No. All shipment, driver, and match data is generated locally. I did not use
real company, customer, or driver information.

## 4. Why did you use synthetic data?

I did not have permission to use real logistics data. Synthetic data let me
build the full pipeline safely and reproducibly. The trade-off is that the
metrics do not establish real-world performance.

## 5. How large is the dataset?

The default generator creates 1,000 shipments, 300 drivers, and 12,000
historical candidate matches. After hard-rule screening, 8,939 candidate rows
reach model training and evaluation.

## 6. What is the target variable?

`match_success` is one when an eligible candidate accepts the shipment,
completes it, stays within the delay threshold, and receives an acceptable
customer rating. Otherwise it is zero.

## 7. Is the target deterministic?

No. The generator uses meaningful logistics relationships plus random and
unobserved effects. A perfectly deterministic target would make the model task
unrealistically easy.

## 8. What are the hard business rules?

The driver must have the exact required vehicle type, enough payload capacity,
and must not be offline. These checks happen before model scoring.

## 9. Why not let the model learn vehicle and capacity compatibility?

Those are operational constraints, not preferences. Treating them as model
features inflated the first evaluation because impossible matches were easy to
reject. Explicit rules are safer and more explainable.

## 10. What features does the model use?

The 12 inputs include pickup-distance score, capacity utilization, normalized
rating, availability, price efficiency, delay and cancellation quality,
response-time score, urgency compatibility, historical success, route
distance, and completed-job experience.

## 11. Are all features available before matching?

They are designed to be available before recommendation in the prototype. In a
real system, historical rates and job counts would need to be calculated as of
the recommendation timestamp to prevent future leakage.

## 12. What is price efficiency?

It is the shipment offer divided by the estimated driver price. A higher value
suggests the offer is more attractive relative to expected cost.

## 13. Why keep capacity utilization after capacity screening?

Two drivers can both have enough capacity but use it differently. Capacity
utilization can represent how well the shipment fits the vehicle rather than
only whether it is technically possible.

## 14. Why use Logistic Regression?

It is an interpretable and well-understood baseline for binary classification.
It helps show whether a nonlinear model adds value beyond a relatively simple
linear decision boundary.

## 15. Why use Random Forest?

Random Forest can capture nonlinear relationships and interactions without
requiring extensive manual transformations. It is also practical for tabular
data and still supports feature-importance inspection.

## 16. Why was Random Forest selected?

Logistic Regression had slightly better validation ROC-AUC, but Random Forest
had better validation ranking metrics. Under the predefined combined score of
70% ROC-AUC and 30% NDCG@5, Random Forest narrowly ranked first.

## 17. Was Random Forest better on every metric?

No. Logistic Regression had higher validation recall, F1, and ROC-AUC. Random
Forest was selected for the combined classification-and-ranking objective. I
would not claim universal superiority.

## 18. How did you split the data?

I split by complete shipment IDs into training, validation, and test sets.
Candidates for one shipment cannot appear in more than one split.

## 19. Why split by shipment rather than candidate row?

Drivers competing for the same shipment share shipment-level information.
Randomly splitting candidate rows could place almost identical shipment
contexts in training and testing and make evaluation too optimistic.

## 20. How is missing data handled?

Missing numerical values flow into the scikit-learn pipeline. A median imputer
is fitted only on the training data. Logistic Regression then applies standard
scaling.

## 21. What data leakage did you check?

I verified that acceptance, completion, delay, customer rating, final outcome
profit, and `match_success` are not model inputs. I also corrected pre-split
imputation and aligned the training population with candidates scored at
inference.

## 22. What was wrong with the first model result?

The original ROC-AUC of 0.967 included many wrong-vehicle or
insufficient-capacity candidates. Those were easy negatives, even though the
live recommender filtered them out. The data generator also allowed some
impossible successes.

## 23. How did you fix the first version?

I made successful labels require hard eligibility, screened the same candidate
population for training and inference, moved imputation into model pipelines,
added validation and untouched test sets, removed duplicate availability
adjustment, and strengthened the tests.

## 24. What is ROC-AUC?

ROC-AUC measures how often the model ranks a random positive above a random
negative across classification thresholds. A value of 0.5 is random ordering,
while 1.0 is perfect separation.

## 25. What is F1-score?

F1 is the harmonic mean of precision and recall at a chosen threshold. It is
useful when both false positives and false negatives matter and the classes
are not perfectly balanced.

## 26. What is Precision@5?

For each shipment, Precision@5 is the proportion of the top five ranked
drivers that are successful historical matches. The reported value is averaged
across shipments.

## 27. What is NDCG@5?

NDCG@5 rewards placing successful drivers nearer the top and normalizes the
score against the best possible ordering for each shipment. It is more
order-sensitive than Precision@5.

## 28. What is top-5 success rate?

It is the proportion of shipments where at least one successful candidate
appears in the first five. It is intuitive, but it can look high when
shipments have several positive candidates, so I do not use it alone.

## 29. What are the final test metrics?

Random Forest achieved accuracy 0.689, precision 0.513, recall 0.586, F1
0.547, ROC-AUC 0.714, Precision@5 0.436, NDCG@5 0.656, and top-5 success
0.947 on 150 untouched shipments.

## 30. Why is the final ROC-AUC only 0.714?

The corrected task is harder because the model ranks only feasible candidates,
which are more similar to each other. The label also contains randomness. I
prefer a moderate defensible result over an inflated synthetic metric.

## 31. How does live recommendation work?

The recommender validates the shipment, creates one candidate row per driver,
calculates pickup distance and estimated cost, applies the shared feature
engineering, removes infeasible drivers, loads the saved pipeline, predicts
probabilities, and sorts them.

## 32. Are the displayed scores true probabilities?

They are model `predict_proba` outputs, but they are not calibrated against real
logistics outcomes. I describe them as prototype match scores rather than
guaranteed operational probabilities.

## 33. How are recommendation explanations generated?

They use actual candidate values, such as matching vehicle, sufficient
capacity, high rating, short pickup distance, low delay rate, reasonable
estimated price, and current availability. They are rule-based explanations,
not SHAP values.

## 34. What does the API provide?

FastAPI exposes health, driver, shipment, metadata, and recommendation
endpoints. Pydantic validates cities, vehicle types, positive numerical
values, urgency options, and route consistency.

## 35. What does the Streamlit demo provide?

It provides a shipment input form, top-five score cards, a score chart, a
candidate table, and plain-language reasons. It is intended for a local
portfolio demonstration.

## 36. What tests did you write?

The tests cover generated schemas, successful-label constraints, engineered
features, missing-value behavior, ranking metric definitions, recommendation
ordering and score bounds, impossible matches, saved artifact metadata, and
API requests.

## 37. Why did you not use deep learning?

The data is structured and modest in size. Simpler models are faster, easier
to debug, and more explainable. I would only add deep learning after
establishing a meaningful baseline and a clear reason.

## 38. Would you use learning-to-rank next?

Yes. The current method is pointwise classification followed by sorting. With
real grouped interaction data, I would compare pairwise or listwise ranking
methods and evaluate them with ranking and business metrics.

## 39. What would production require?

Real timestamped data, point-in-time features, robust data validation,
calibrated probabilities, security, scalable serving, monitoring, drift
detection, fairness review, retraining, and controlled online experiments.

## 40. What did you personally learn?

I learned that model quality depends on candidate definition and evaluation
design, not only model choice. I also learned to separate hard rules from
learned ranking, contain preprocessing inside pipelines, preserve a true test
set, and communicate limitations honestly.
