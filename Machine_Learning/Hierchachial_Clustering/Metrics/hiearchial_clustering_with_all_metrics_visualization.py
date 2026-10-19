import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering

# Create sample data
X, _ = make_blobs(n_samples=50, centers=3, random_state=42)

# Try with different distance metrics
for metric in ["euclidean", "manhattan", "cosine"]:
    clustering = AgglomerativeClustering(
        n_clusters=3,
        metric=metric,  # distance metric
        linkage="average"  # linkage criterion
    )
    labels = clustering.fit_predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="rainbow")
    plt.title(f"Agglomerative Clustering ({metric} distance)")
    plt.show()
