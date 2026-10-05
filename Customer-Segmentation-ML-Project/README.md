# Customer Segmentation with Clustering

An unsupervised-learning project that groups customers by purchasing behavior. The notebook compares centroid-based, hierarchical, and density-based clustering, then interprets the resulting customer segments.

## Workflow

- Explore age, annual income, purchase amount, loyalty score, region, and purchase frequency.
- Select behavioral numeric features and standardize them.
- Use the elbow method and silhouette score to choose a K-means cluster count.
- Fit a four-cluster K-means model and profile each segment.
- Project the standardized data to two dimensions with PCA for visualization.
- Build a hierarchical-clustering dendrogram.
- Test DBSCAN at several neighborhood radii to study clusters and noise points.
- Export K-means, two DBSCAN variants, and the fitted scaler.

## Notebook walkthrough

1. Inspect customer demographics and purchasing measures, check duplicates, and plot distributions and relationships among income, spending, loyalty, and purchase frequency.
2. Remove `user_id` and `region` from the clustering feature matrix, then standardize the remaining numeric features. Region is retained only for later profile comparisons.
3. Examine K-means inertia for 1–10 clusters and silhouette scores for 2–10 clusters. Fit a four-cluster K-means model with `random_state=42` and `n_init=10`.
4. Profile the four groups by size, mean numeric features, and region mix; project the standardized matrix to two PCA dimensions for a visual check.
5. Compare Ward hierarchical clustering at four groups and a dendrogram with K-means assignments. Use a five-nearest-neighbor distance plot to explore DBSCAN radius choices.
6. Try DBSCAN with `min_samples=10` and `eps` values 0.18, 0.30, and 0.45, counting assigned clusters and noise points. Export the 0.30 and 0.45 models along with K-means and the scaler.

## Dataset

`Customer Purchasing Behaviors.csv` contains 237 customer records. Its fields include user ID, age, annual income, purchase amount, loyalty score, region, and purchase frequency.

## Repository contents

| File | Purpose |
| --- | --- |
| `Customer-Segmentation.ipynb` | Exploration, clustering comparisons, visualization, and segment analysis |
| `Customer Purchasing Behaviors.csv` | Source customer dataset |
| `kmeans_model.pkl` | Exported four-cluster K-means model |
| `dbscan_model.pkl` | Exported DBSCAN model with a moderate radius |
| `dbscan_model_loose.pkl` | Exported DBSCAN model with a looser radius |
| `scaler.pkl` | Exported standard scaler |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install jupyter numpy pandas matplotlib scipy scikit-learn joblib
jupyter lab Customer-Segmentation.ipynb
```

Cluster numbers are arbitrary identifiers. Interpret them from the notebook's per-cluster feature summaries rather than treating the numeric label as an ordered score.
