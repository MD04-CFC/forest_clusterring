import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

from sklearn.metrics import confusion_matrix
import pandas as pd



# Tworzenie macierzy kontyngencji


# Load dataset
df = pd.read_csv("dataset.csv")  # Replace with actual file path
df = df.sample(frac=0.2, random_state=42)





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
dbscan = DBSCAN(eps=0.8, min_samples=13)  # Adjust parameters as needed
dbscan_labels = dbscan.fit_predict(X_scaled)

true_labels = df["Cover_Type"] 
conf_matrix = pd.crosstab(true_labels, dbscan_labels, rownames=['Rzeczywiste'], colnames=['DBSCAN'])

print(conf_matrix)

# Apply Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=7)
hierarchical_labels = hierarchical.fit_predict(X_scaled)

# Evaluation function
def evaluate_clustering(X, labels, method):
    if len(set(labels)) > 1:
        silhouette = silhouette_score(X, labels)
        db_index = davies_bouldin_score(X, labels)
        print(f"{method}: Silhouette Score = {silhouette:.3f}, Davies-Bouldin Index = {db_index:.3f}")
    else:
        print(f"{method}: Too few clusters to compute evaluation metrics.")

#evaluate_clustering(X_scaled, kmeans_labels, "K-Means")
#evaluate_clustering(X_scaled, dbscan_labels, "DBSCAN")
#evaluate_clustering(X_scaled, hierarchical_labels, "Hierarchical Clustering")