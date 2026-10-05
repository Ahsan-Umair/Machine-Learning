# Customer Churn Predictor

A supervised machine-learning project for predicting whether a customer will churn from demographic, account, usage, support, payment, and subscription information. The notebook emphasizes both predictive performance and the effect of a potentially dominant feature.

## Workflow

- Inspect and clean the customer dataset.
- Explore churn balance and relationships between churn and customer attributes.
- Encode categorical variables and standardize numeric features.
- Train and compare logistic-regression and random-forest classifiers.
- Remove `Payment Delay` in an ablation experiment to measure its influence.
- Tune the random forest with cross-validated grid search using F1 score.
- Export the selected random forest and the logistic-regression scaler as separate artifacts.

## Notebook walkthrough

1. Inspect duplicates, missing values, feature correlations, category levels, and the churn rate. Plot numeric distributions and compare churn rates by gender, contract length, and subscription type.
2. Remove `CustomerID`, encode gender as 0/1, and one-hot encode contract and subscription categories with one reference level dropped.
3. Make a stratified 80/20 train/test split with `random_state=42`. Fit `StandardScaler` only on the training features used by logistic regression; train the forest on the unscaled encoded table.
4. Compare logistic regression and a random forest using accuracy, precision, recall, F1, and confusion matrices. Rank forest feature importances.
5. Retrain the forest after removing `Payment Delay` and compare its scores with the full-feature models. This ablation tests dependence on a particularly strong predictor.
6. Tune the forest with five-fold F1 grid search over tree count, depth, minimum split size, and minimum leaf size. Evaluate the selected model and save it as `churn_model_rf.pkl`.

The saved forest needs the same encoded columns and order as training, but it was **not** trained on standardized values. `churn_scaler.pkl` was fitted for the logistic-regression baseline.

## Dataset

`customer_churn_dataset-testing-master.csv` contains 64,374 customer records. Features include age, gender, tenure, usage frequency, support calls, payment delay, subscription type, contract length, total spend, and last interaction. The target column is `Churn`.

## Repository contents

| File | Purpose |
| --- | --- |
| `Customer_Churn_Predictor.ipynb` | Exploration, preprocessing, model comparison, tuning, and export |
| `customer_churn_dataset-testing-master.csv` | Source dataset |
| `churn_model_rf.pkl` | Exported random-forest classifier |
| `churn_scaler.pkl` | Exported feature scaler |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install jupyter numpy pandas matplotlib scikit-learn joblib
jupyter lab Customer_Churn_Predictor.ipynb
```

Run all cells in order. The exported forest needs the same categorical encoding and feature order used for training; it does not use the saved scaler.
