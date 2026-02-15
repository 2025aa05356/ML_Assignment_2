1. probelm statement-Wine Quality Prediction ML Models
2. Dataset
	•	Instances: 1599
	•	Features: 12
	•	Classes: 0–5 (mapping: {3:0,4:1,5:2,6:3,7:4,8:5})
	•	Train/Test Split: 1279 / 320
3. model used and their evaluations:

| Model               | Accuracy | AUC    | Precision | Recall | F1    | MCC   |
|--------------------|---------|--------|-----------|--------|-------|-------|
| Logistic Regression | 0.591   | 0.764  | 0.570     | 0.591  | 0.567 | 0.325 |
| Decision Tree       | 0.609   | 0.658  | 0.612     | 0.609  | 0.609 | 0.398 |
| KNN                 | 0.609   | 0.698  | 0.584     | 0.609  | 0.596 | 0.373 |
| Naive Bayes         | 0.562   | 0.684  | 0.574     | 0.562  | 0.568 | 0.330 |
| Random Forest       | 0.675   | 0.766  | 0.650     | 0.675  | 0.660 | 0.477 |
| XGBoost             | 0.653   | 0.799  | 0.648     | 0.653  | 0.643 | 0.445 |

Observations
| Model               | Observation about model performance                       |
|--------------------|-----------------------------------------------------------|
| Logistic Regression | Moderate accuracy; decent AUC; struggles with precision. |
| Decision Tree       | Slightly better accuracy; may overfit training data.     |
| KNN                 | Similar accuracy to Decision Tree; good recall.          |
| Naive Bayes         | Lowest accuracy; struggles with correlated features.     |
| Random Forest       | Best overall performance; balanced precision and recall.|
| XGBoost             | High AUC; performs well on imbalanced data; good F1.    |