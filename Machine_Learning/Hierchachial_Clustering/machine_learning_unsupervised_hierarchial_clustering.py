import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


df = pd.read_csv("Mall_Customers.csv")

# Select relevant features
X = df[["Annual Income (k$)", "Spending Score (1–100)"]]

# Step 1: Plot dendrogram
linked = linkage(X, method="ward")

plt.figure(figsize=(10, 5))
dendrogram(linked, truncate_mode="lastp", p=20, leaf_rotation=45, leaf_font_size=10)
plt.title("Dendrogram for Mall Customers")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")
plt.show()

# Step 2: Fit Hierarchical Clustering (decide n_clusters from dendrogram, say 5)
hc = AgglomerativeClustering(n_clusters=5, metric="euclidean", linkage="ward")
df["Cluster"] = hc.fit_predict(X)

# Step 3: Visualize clusters
plt.figure(figsize=(8, 6))
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=df["Cluster"], cmap="rainbow")
plt.title("Customer Segments")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1–100)")
plt.show()

print(df.head())
