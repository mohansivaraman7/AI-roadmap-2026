import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import BisectingKMeans

# Load dataset (Mall_Customers.csv)
df = pd.read_csv("Mall_Customers.csv")

# Select features
X = df[["Annual Income (k$)", "Spending Score (1–100)"]]

# Divisive Clustering using Bisecting KMeans
div_cluster = BisectingKMeans(n_clusters=5, random_state=42)
df["DivisiveCluster"] = div_cluster.fit_predict(X)

# Visualize clusters
plt.figure(figsize=(8, 6))
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=df["DivisiveCluster"], cmap="rainbow")
plt.title("Divisive (Bisecting KMeans) Customer Segments")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1–100)")
plt.show()

print(df.head())
