# Breast Cancer Predictor

An end-to-end binary classification project that predicts whether a breast mass is benign or malignant from diagnostic measurements. The notebook covers exploratory analysis, preprocessing, model comparison, feature importance, feature ablation, hyperparameter tuning, and model export.

> This project is for educational purposes only. It is not a medical device and must not be used for diagnosis or treatment decisions.

## Workflow

- Load and inspect the Wisconsin-style breast cancer dataset.
- Clean identifiers and encode the diagnosis target.
- Explore class balance, feature distributions, correlations, and outliers.
- Split the data into training and test sets and standardize numeric features.
- Compare logistic regression with random forest classification.
- Inspect random-forest feature importance and test an ablated feature set.
- Tune the random forest with cross-validated grid search using F1 score.
- Save the fitted classifier and scaler with Joblib.

## Notebook walkthrough

1. Inspect the 569 records for missing values, duplicates, ID reuse, class balance, and feature types. Drop `id`, then map malignant (`M`) to 1 and benign (`B`) to 0.
2. Plot the diagnosis counts and measurement distributions, then rank correlations with the target. These plots support exploration; they do not select features for the baseline models.
3. Make a stratified 80/20 train/test split with `random_state=42`. Fit `StandardScaler` on training features only for logistic regression. Train the random forest on the original numeric feature values.
4. Evaluate both classifiers on the held-out test set with accuracy, precision, recall, F1, and confusion matrices. Plot the forest's feature importances.
5. Repeat the forest experiment after removing six mean/worst radius, perimeter, and area features to see how much those correlated measurements matter.
6. Search forest sizes of 100, 200, and 300 trees, maximum depths of none/5/10/15, and minimum split sizes of 2/5/10 with five-fold F1 cross-validation. Re-evaluate the selected forest on the test set.
7. Save the tuned forest and the scaler as separate artifacts, then reload the forest for sample predictions. The saved forest expects the original unscaled measurement columns in training order; the scaler belongs to the logistic-regression experiment.

## Dataset

`breast-cancer.csv` contains 569 observations. The target is `diagnosis`, and the predictors are measurements such as radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.

## Repository contents

| File | Purpose |
| --- | --- |
| `Breast-Cancer-Predictor.ipynb` | Complete analysis, training, evaluation, and export workflow |
| `breast-cancer.csv` | Source dataset |
| `breast_cancer_rf_model.joblib` | Exported tuned random-forest classifier |
| `breast_cancer_scaler.joblib` | Exported standard scaler |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install jupyter numpy pandas matplotlib scikit-learn joblib
jupyter lab Breast-Cancer-Predictor.ipynb
```

Run the notebook from top to bottom to reproduce the preprocessing, evaluation, and exported artifacts. Serialized models should be loaded with compatible versions of scikit-learn and Joblib.
