import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Load Iris dataset
iris = load_iris()
X = iris.data

# Reduce to 2D for visualization using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

linkage_methods = ['single', 'complete', 'average', 'ward']

plt.figure(figsize=(14, 10))

for i, method in enumerate(linkage_methods):
    plt.subplot(2, 2, i+1)
    Z = linkage(X, method=method)
    dendrogram(Z, truncate_mode='level', p=3)
    plt.title(f'Dendrogram ({method.title()} Linkage)')
    plt.xlabel('Sample Index or Cluster Size')
    plt.ylabel('Distance')

plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 10))
for i, method in enumerate(linkage_methods):
    plt.subplot(2, 2, i+1)
    if method == 'ward':
        clusterer = AgglomerativeClustering(n_clusters=3, linkage=method)
    else:
        clusterer = AgglomerativeClustering(n_clusters=3, linkage=method, affinity='euclidean')
    labels = clusterer.fit_predict(X)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    plt.title(f'Clusters ({method.title()} Linkage)')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')

plt.tight_layout()
plt.show()