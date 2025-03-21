import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score


# Load dataset
df = pd.read_csv("output.csv")  # Replace with actual file path
df = df.sample(frac=0.05, random_state=42)


# MINIREADME: I don't know how on your device, but on mine fetching data using fetch_ucirepo was too slow, that's why I used the csv file here, which I added to the github last week


# Define features (excluding binary variables, but keeping Cover_Type for analysis)
features = [
    "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points"
]



X = df[features]
#print(X)

y = df["Cover_Type"]  # Keep this for analysis, but don't use for clustering


# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#print(X_scaled)

# Apply K-Means
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
#print(kmeans_labels)

# Apply DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=10)  # Adjust parameters as needed
dbscan_labels = dbscan.fit_predict(X_scaled)

# Apply Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=5)
hierarchical_labels = hierarchical.fit_predict(X_scaled)

# Evaluation function
def evaluate_clustering(X, labels, method):
    if len(set(labels)) > 1:
        silhouette = silhouette_score(X, labels)
        db_index = davies_bouldin_score(X, labels)
        print(f"{method}: Silhouette Score = {silhouette:.3f}, Davies-Bouldin Index = {db_index:.3f}")
    else:
        print(f"{method}: Too few clusters to compute evaluation metrics.")

evaluate_clustering(X_scaled, kmeans_labels, "K-Means")
evaluate_clustering(X_scaled, dbscan_labels, "DBSCAN")
evaluate_clustering(X_scaled, hierarchical_labels, "Hierarchical Clustering")

# Analyzing how clusters relate to Cover_Type
df["KMeans_Cluster"] = kmeans_labels
df["DBSCAN_Cluster"] = dbscan_labels
df["Hierarchical_Cluster"] = hierarchical_labels

# Cross-tabulation to check Cover_Type distribution within K-Means clusters
print("\nCover Type Distribution in K-Means Clusters:")
print(pd.crosstab(df["KMeans_Cluster"], df["Cover_Type"], normalize="index"))

# Visualization (K-Means example using PCA)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df["KMeans_Cluster"], palette="viridis")
plt.title("K-Means Clustering (PCA Reduced)")
plt.show()
