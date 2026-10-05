# Hidden Market Factors with PCA

An unsupervised-learning project that applies principal component analysis to stock returns to uncover shared market factors. It transforms long-form daily stock prices into a return matrix, studies explained variance, and interprets stock loadings on the leading components.

## Workflow

- Inspect five years of daily OHLCV data for multiple stocks.
- Pivot closing prices into a date-by-stock matrix.
- Calculate returns and handle incomplete observations.
- Standardize stock return series before PCA.
- Use explained-variance ratios and a scree plot to select four components.
- Analyze component scores and per-stock loadings.
- Visualize stocks in principal-component space, including sector-oriented comparisons.
- Export the fitted scaler and four-component PCA model.

## Notebook walkthrough

1. Inspect five years of stock rows, duplicate and missing data, ticker formatting, date coverage, and the number of names. Drop incomplete source rows.
2. Pivot daily closing prices into a date-by-ticker matrix, calculate percentage returns, remove the first undefined return row, and drop tickers with remaining missing returns.
3. Standardize each retained ticker's return series. Fit an unrestricted PCA to inspect individual and cumulative explained variance, then plot the first 20 components as a scree chart.
4. Fit a separate four-component PCA and examine each ticker's loadings on PC1–PC4, listing the strongest positive and negative contributors.
5. Plot stocks by their first two loadings and highlight handpicked utilities and bank tickers to explore a possible sector pattern. Those factor labels are interpretations, not guaranteed economic causes.
6. Save the scaler and four-component PCA model. New input needs the same ticker universe, order, return calculation, and normalization.

## Dataset

`all_stocks_5yr.csv` contains 619,040 daily stock rows with date, open, high, low, close, volume, and ticker name fields.

## Repository contents

| File | Purpose |
| --- | --- |
| `Hidden-Market-Factor.ipynb` | Return preparation, PCA fitting, interpretation, and visualization |
| `all_stocks_5yr.csv` | Five-year stock-market dataset |
| `market_factor_pca.pkl` | Exported four-component PCA model |
| `market_factor_scaler.pkl` | Exported return scaler |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install jupyter numpy pandas matplotlib scikit-learn joblib
jupyter lab Hidden-Market-Factor.ipynb
```

The discovered components are statistical factors, not guaranteed economic or causal factors. This project is for research and education and does not provide investment advice.
