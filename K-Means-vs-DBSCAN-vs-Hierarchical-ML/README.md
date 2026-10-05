# K-Means vs DBSCAN vs Hierarchical Clustering

A side-by-side clustering study on a nonlinear two-moons dataset. It shows how algorithm assumptions affect the discovered groups and compares results against known synthetic labels.

## Workflow

- Generate noisy two-moons data with scikit-learn.
- Standardize the two input dimensions.
- Select K for K-means with the elbow method and fit a two-cluster model.
- Build a single-linkage dendrogram and compare single- and average-linkage agglomerative clustering.
- Fit DBSCAN at multiple `eps` values to study its treatment of curved clusters and noise.
- Visualize the three main clustering results side by side.
- Compare adjusted Rand index against the known labels and silhouette score within each predicted partition.
- Export the selected DBSCAN model and scaler.

## Notebook walkthrough

1. Generate 300 noisy two-moons samples with `noise=0.08` and `random_state=42`; plot them both without labels and with the known synthetic labels for reference.
2. Standardize the two coordinates, inspect K-means inertia for cluster counts 1–9, and fit a two-cluster K-means model.
3. Build a single-linkage dendrogram, then fit two-group agglomerative clustering with both single and average linkage to compare their partitions.
4. Fit DBSCAN with `min_samples=5` at `eps=0.30`, then repeat at `eps=0.15` to see how the tighter radius changes groups and noise labels (`-1`).
5. Plot K-means, hierarchical, and DBSCAN results side by side. Calculate adjusted Rand index against the known moon labels and silhouette score from the predicted partitions.
6. Export the `eps=0.30` DBSCAN estimator and the fitted scaler. A new point can be scaled the same way, but scikit-learn's fitted DBSCAN object does not provide a normal `predict` method for assigning arbitrary new points.

## Repository contents

| File | Purpose |
| --- | --- |
| `Comparison.ipynb` | Data generation, clustering, visualization, and evaluation |
| `dbscan_model.pkl` | Exported DBSCAN model |
| `scaler.pkl` | Exported standard scaler |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install jupyter numpy matplotlib scipy scikit-learn joblib
jupyter lab Comparison.ipynb
```

The notebook is designed to demonstrate that internal metrics and agreement with known labels measure different properties. A visually appropriate nonlinear partition may not rank identically under every metric.
