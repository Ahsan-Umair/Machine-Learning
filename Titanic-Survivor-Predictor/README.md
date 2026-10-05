# Titanic Survivor Predictor

A classification project that predicts passenger survival from the Titanic passenger manifest. The notebook covers exploratory analysis, missing-value handling, feature engineering, model comparison, cross-validation, hyperparameter tuning, and reusable model export.

## Workflow

- Explore survival by sex, passenger class, age, fare, and family composition.
- Fill missing embarkation and age values with group-aware strategies.
- Engineer cabin availability, title, family size, and solo-traveler features.
- Encode categorical variables and standardize model inputs.
- Compare logistic regression, decision tree, random forest, SVM, and KNN.
- Validate the candidates with five-fold cross-validation.
- Tune random forest and SVM with grid search and evaluate the best on a held-out test set.
- Export the selected classifier and standard scaler.

## Notebook walkthrough

1. Explore survival by sex, passenger class, age, fare, and family composition. Fill missing embarkation ports with the mode and missing ages with medians grouped by passenger class and sex.
2. Engineer `Has Cabin`, honorific `Titles`, `Family Size`, and `IsAlone`; encode sex, embarkation port, and title, then remove identifiers and free-text ticket/name/cabin fields.
3. Split the processed table into training and test sets, and fit a scaler for models that use standardized inputs.
4. Compare logistic regression, decision tree, random forest, support-vector machine, and K-nearest neighbors. Plot training versus test accuracy, inspect the best candidate's confusion matrix and classification report, and run five-fold cross-validation.
5. Grid-search forest and SVM settings, compare their held-out accuracies, and choose the better of those tuned estimators as `final_model`.
6. Save `final_model` and the scaler. New passengers still need the same imputation, engineered fields, one-hot columns, and feature order before prediction.

The final choice between tuned forest and SVM uses test-set accuracy, so those test scores also participate in model selection. A separate untouched evaluation set would be needed for an unbiased final estimate.

## Dataset

`Titanic-Dataset.csv` contains 891 passenger records. The target is `Survived`; the source fields include ticket class, name, sex, age, siblings/spouses, parents/children, ticket, fare, cabin, and embarkation port.

## Repository contents

| File | Purpose |
| --- | --- |
| `Titanic-Predictor.ipynb` | Full feature-engineering and model-selection workflow |
| `Titanic-Dataset.csv` | Passenger dataset |
| `titanic_model.pkl` | Exported final classifier |
| `titanic_scaler.pkl` | Exported standard scaler |
| `PROJECT_NOTEBOOK_EXPLAINED_FOR_5YO.txt` | Plain-language walkthrough of the notebook |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install jupyter numpy pandas matplotlib scikit-learn joblib
jupyter lab Titanic-Predictor.ipynb
```

New inference records must reproduce the notebook's feature engineering and final encoded column order before scaling and prediction.
