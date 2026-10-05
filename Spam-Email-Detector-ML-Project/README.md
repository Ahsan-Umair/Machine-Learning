# Spam Email Detector

A natural-language classification project that labels email text as spam or legitimate mail. The notebook covers text cleaning, exploratory word analysis, TF-IDF vectorization, leakage checks, classifier comparison, hyperparameter tuning, and artifact export.

## Workflow

- Remove duplicate messages and inspect class balance and email length.
- Normalize text with lowercasing, non-letter removal, and whitespace cleanup.
- Compare common words in spam and non-spam messages.
- Split the messages with stratification and convert text to TF-IDF features.
- Train logistic-regression and random-forest classifiers.
- Run an ablation experiment without highly predictive leakage-like words.
- Tune the random forest using cross-validated F1 score.
- Evaluate accuracy, precision, recall, F1, classification reports, and confusion matrices.
- Export the final classifier and fitted TF-IDF vectorizer.

## Notebook walkthrough

1. Remove duplicate emails, inspect missing values and message lengths, and compare the spam proportion with a majority-class accuracy baseline.
2. Lowercase text, remove non-letter characters, collapse whitespace, and inspect frequent words separately in spam and legitimate messages. Plot message word counts by class.
3. Stratify the train/test split and fit a TF-IDF vectorizer on training text only, using English stop words and at most 5,000 features.
4. Fit logistic regression, inspect its strongest positive and negative word coefficients, and evaluate accuracy, precision, recall, F1, confusion matrix, and classification report.
5. Repeat a text-feature experiment while removing dataset-specific words such as `enron`, `vince`, `ect`, and `kaminski` to check how much those cues drive classification.
6. Compare a random forest, inspect word importances, tune its hyperparameters with F1 grid search, and save the selected forest plus the original fitted TF-IDF vectorizer.

Inference must apply the notebook's text cleaner before the saved vectorizer. The ablation vectorizer is an experiment and is not the exported vectorizer.

## Dataset

`emails 2.csv` contains 5,728 rows with two columns: raw email `text` and the binary `spam` label.

## Repository contents

| File | Purpose |
| --- | --- |
| `Spam-Email-Detector.ipynb` | Text analysis, feature extraction, training, tuning, and export |
| `emails 2.csv` | Labeled email dataset |
| `spam_classifier_rf.pkl` | Exported random-forest classifier |
| `tfidf_vectorizer.pkl` | Exported TF-IDF vectorizer |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install jupyter numpy pandas matplotlib scikit-learn joblib
jupyter lab Spam-Email-Detector.ipynb
```

Inference must clean text in the same way as the notebook and transform it with `tfidf_vectorizer.pkl` before classification.
