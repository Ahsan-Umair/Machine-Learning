# Credit Card Fraud Anomaly Detection

An unsupervised anomaly-detection project that uses Isolation Forest to flag unusual credit-card transactions. Fraud labels are retained for evaluation, but they are not used to train the anomaly detector.

## Workflow

- Inspect transaction amount, time, class imbalance, and feature distributions.
- Derive `Hour` from `Time`, drop `Time`, and robust-scale the remaining numeric features.
- Train an Isolation Forest with a contamination rate informed by fraud prevalence.
- Convert anomaly predictions into the dataset's normal/fraud label convention.
- Evaluate with a classification report and confusion matrix.
- Compare anomaly-score distributions for normal and fraudulent transactions.
- Export the fitted Isolation Forest and robust scaler.

## Notebook walkthrough

1. Inspect the credit-card transaction table, amount/time distributions, anonymized `V` features, duplicate rows, missing values, zero amounts, and the strong imbalance of fraud labels.
2. Drop exact duplicates. Hold `Class` aside as an evaluation answer key; it is not a model input. Derive an `Hour` feature from `Time` and then drop the original `Time` column.
3. Fit `RobustScaler` on **all remaining numeric features**, including amount and hour, then fit a 100-tree Isolation Forest with `random_state=42`. Its contamination parameter is set to the observed fraud fraction.
4. Convert Isolation Forest's `-1` anomaly output to the fraud class (`1`), compare it with the retained labels through a confusion matrix and classification report, and inspect anomaly-score distributions and caught versus missed fraud cases.
5. Save the detector and scaler separately. Reuse the same duplicate policy, `Hour` derivation, feature order, and scaling at inference time.

The notebook scores the same transactions used to fit the unsupervised detector, so its classification report is descriptive rather than an independent test estimate. The class labels also influence the contamination setting, even though they are excluded from the fitted feature matrix.

## Dataset

`creditcard.csv` contains 284,807 transactions. Most predictors are anonymized PCA components (`V1` through `V28`), accompanied by `Time`, `Amount`, and the evaluation label `Class`.

## Repository contents

| File | Purpose |
| --- | --- |
| `AnomalyDetection.ipynb` | Exploration, preprocessing, anomaly detection, and evaluation |
| `creditcard.csv` | Transaction dataset |
| `fraud_isolation_forest.joblib` | Exported Isolation Forest |
| `fraud_scaler.joblib` | Exported robust scaler |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install jupyter numpy pandas matplotlib seaborn scikit-learn joblib
jupyter lab AnomalyDetection.ipynb
```

This is an educational experiment, not a production fraud-control system. Real deployment requires threshold calibration, temporal validation, monitoring, investigation workflows, and cost-sensitive evaluation.
