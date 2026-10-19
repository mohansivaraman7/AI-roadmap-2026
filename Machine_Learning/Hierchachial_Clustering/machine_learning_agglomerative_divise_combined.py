import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import BisectingKMeans

# Generate synthetic dataset
X, _ = make_blobs(n_samples=20, centers=3, cluster_std=1.2, random_state=42)

# =======================================================
# 1. Agglomerative Hierarchical Clustering
# =======================================================
methods = ["single", "complete", "average", "ward"]

plt.figure(figsize=(14, 10))
for i, method in enumerate(methods, 1):
    plt.subplot(2, 2, i)
    Z = linkage(X, method=method)   # Perform clustering
    dendrogram(Z, labels=[f"P{j}" for j in range(1, len(X)+1)])
    plt.title(f"Agglomerative - {method.capitalize()} Linkage")
    plt.xlabel("Data Points")
    plt.ylabel("Distance")

plt.tight_layout()
plt.show()

# =======================================================
# 2. Divisive Hierarchical Clustering (using Bisecting KMeans)
# =======================================================
# We’ll split into 3 clusters for illustration
divisive = BisectingKMeans(n_clusters=3, random_state=42)
labels = divisive.fit_predict(X)

plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="rainbow", s=60, edgecolor="k")
plt.title("Divisive Clustering (Bisecting KMeans)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
