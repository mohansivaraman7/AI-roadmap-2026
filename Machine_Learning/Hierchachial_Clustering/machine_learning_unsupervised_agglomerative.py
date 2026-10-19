from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Create synthetic data
X, y_true = make_blobs(n_samples=200, centers=3, cluster_std=0.6, random_state=0)

# Agglomerative clustering
model = AgglomerativeClustering(n_clusters=3, linkage='ward', metric="euclidean")
labels = model.fit_predict(X)

# Simple visualization
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30)
plt.title('AgglomerativeClustering (Ward, Euclidean)')
plt.show()
