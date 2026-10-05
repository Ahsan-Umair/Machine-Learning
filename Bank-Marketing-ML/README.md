# Bank Marketing Classification

A machine-learning study of whether a bank client will subscribe to a term deposit after a marketing campaign. The notebook focuses on class imbalance, leakage-aware feature selection, reliable cross-validation, regularized logistic regression, and an end-to-end preprocessing pipeline.

## Workflow

- Load the semicolon-delimited Bank Marketing dataset and inspect its distributions.
- Explore campaign outcomes across client and contact attributes.
- Prepare numeric and categorical columns with a `ColumnTransformer`.
- Compare ordinary K-fold and stratified K-fold validation.
- Compare baseline, L1-regularized, and L2-regularized logistic regression.
- Tune regularization strength with grid and randomized search using F1 score.
- Evaluate the selected pipeline with accuracy, precision, recall, F1, a classification report, and confusion matrix.
- Export one pipeline containing both preprocessing and classification.

## Notebook walkthrough

1. Load `Bank Marketing ML/bank-full.csv` using its semicolon delimiter; inspect class imbalance, the literal `unknown` category, numeric distributions, categorical conversion rates, and subscription rate by month.
2. Derive `was_contacted_before` from the `pdays` sentinel and replace `-1` with 0. Drop call `duration` before modeling because its value would not be known when predicting before a call.
3. In the first experiment, one-hot encode categories, scale numeric fields from the training split, and compare ordinary K-fold against stratified K-fold F1 scores for logistic regression.
4. Compare L1 and L2 regularization and search the regularization strength `C` with both grid and randomized search.
5. Rebuild the final approach as a `ColumnTransformer` plus logistic-regression `Pipeline`, so scaling and one-hot encoding are learned inside cross-validation rather than before it.
6. Select the best pipeline, evaluate probabilities and labels on the held-out test data, inspect the confusion matrix and signed feature coefficients, then save and reload the full pipeline for a sample customer.

## Dataset

The `Bank Marketing ML/` directory includes full and reduced variants of the UCI-style Bank Marketing data, together with their field descriptions. The notebook trains on `Bank Marketing ML/bank-full.csv`, which contains 45,211 client-contact records and a binary subscription target named `y`.

## Repository contents

| Path | Purpose |
| --- | --- |
| `Bank Marketing ML/Bank.ipynb` | Exploration, validation experiments, tuning, and final evaluation |
| `Bank Marketing ML/bank-full.csv` | Full dataset used by the notebook |
| `Bank Marketing ML/bank.csv` | Smaller dataset variant |
| `Bank Marketing ML/bank-additional*.csv` | Alternative dataset variants with economic indicators |
| `Bank Marketing ML/*-names.txt` | Dataset descriptions |
| `Bank Marketing ML/bank_marketing_logistic_pipeline.pkl` | Exported preprocessing and logistic-regression pipeline |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install jupyter numpy pandas matplotlib scikit-learn joblib
jupyter lab "Bank Marketing ML/Bank.ipynb"
```

Run all cells in order. Because preprocessing is embedded in the exported pipeline, pass inference data with the original raw feature names and formats.
