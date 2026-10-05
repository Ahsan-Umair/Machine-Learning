# Student Performance Predictor

A regression project that predicts a student's total score from weekly self-study hours, attendance percentage, and class participation. It includes a transparent linear-regression baseline, model comparison, and exploratory visualizations.

## What is included

- A one-million-row synthetic student-performance dataset.
- A reproducible train/test split.
- Linear-regression coefficients and sample predictions.
- Evaluation with MAE, MSE, RMSE, and R-squared.
- Comparisons with decision trees across multiple depths, KNN, SVR, and random forest.
- Scatter plots for each input feature against total score.

## Repository contents

| File | Purpose |
| --- | --- |
| `main.py` | Trains and evaluates the linear-regression baseline |
| `compare_models.py` | Evaluates linear, tree, neighbor, kernel, and ensemble regressors |
| `visualization.py` | Plots sampled feature/target relationships |
| `student_performance.csv` | Source dataset with 1,000,000 records |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib scikit-learn
python main.py
python compare_models.py
python visualization.py
```

The visualization script samples 5,000 records for responsive plotting. Model evaluation still uses the full dataset and can require noticeably more memory and compute.
