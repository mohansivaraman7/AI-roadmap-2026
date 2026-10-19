import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Create sample data
X, _ = make_blobs(n_samples=50, centers=3, random_state=42)

results = []

for metric in ["euclidean", "manhattan", "cosine"]:
    clustering = AgglomerativeClustering(
        n_clusters=3,
        metric=metric,
        linkage="average"
    )
    labels = clustering.fit_predict(X)

    silhouette = silhouette_score(X, labels, metric=metric)
    dbi = davies_bouldin_score(X, labels)
    chi = calinski_harabasz_score(X, labels)

    results.append({
        "Metric": metric,
        "Silhouette Score": round(silhouette, 3),
        "Davies–Bouldin Index": round(dbi, 3),
        "Calinski–Harabasz Score": round(chi, 3)
    })

import pandas as pd

results_df = pd.DataFrame(results)
print(results_df)
