# Loan Approval Predictor

A binary-classification project that predicts whether a loan application will be approved from applicant finances, assets, requested terms, and credit history. The notebook also performs a feature-ablation experiment to show how strongly the result depends on the CIBIL score.

## Workflow

- Clean column names and inspect applicant, loan, and approval distributions.
- Encode categorical fields and split the data into training and test sets.
- Standardize features for logistic regression.
- Compare logistic regression with random forest using accuracy, precision, recall, F1, and confusion matrices.
- Tune random-forest hyperparameters with cross-validated grid search.
- Retrain without CIBIL score and compare performance to expose feature dependence.
- Export the best tuned random-forest classifier.

## Notebook walkthrough

1. Strip whitespace from column names and category values, drop `loan_id`, and inspect the approval balance. Plot CIBIL score, income, education, employment, and asset-value relationships with approval.
2. Encode education, self-employment, and approval status numerically. Make a stratified 80/20 train/test split with `random_state=42`.
3. Standardize training features for logistic regression and train a 100-tree random forest on the unscaled table. Compare accuracy, precision, recall, F1, and both confusion matrices.
4. Tune the forest with five-fold grid search scored by F1, then measure the chosen estimator on the held-out test set. Save and reload that tuned classifier with Joblib.
5. Rank forest feature importances and repeat the training experiment without `cibil_score` to show how strongly predictions depend on credit score.

The exported forest expects the encoded feature columns in the same order as training. The notebook's scaler is for the logistic-regression comparison and is not exported with the forest.

## Dataset

`loan_approval_dataset.csv` contains 4,269 applications. It includes dependents, education, employment status, annual income, loan amount and term, CIBIL score, residential and commercial assets, luxury assets, bank assets, and the `loan_status` target.

## Repository contents

| File | Purpose |
| --- | --- |
| `Loan_Predictor_Model.ipynb` | Complete exploration, training, tuning, interpretation, and export workflow |
| `loan_approval_dataset.csv` | Source application dataset |
| `loan_approval_model.pkl` | Exported tuned random-forest classifier |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install jupyter numpy pandas matplotlib scikit-learn joblib
jupyter lab Loan_Predictor_Model.ipynb
```

This project is an educational model, not a production lending decision system. Real lending systems require fairness testing, explainability, monitoring, policy review, and compliance controls.
