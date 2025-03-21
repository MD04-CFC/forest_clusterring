# 2D ok, correct clusters, a bit of AI help


from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
  
# fetch dataset 
covertype = fetch_ucirepo(id=31) 
  
# data (as pandas dataframes) 
X2 = covertype.data.features 
y = covertype.data.targets 
X = X2[['Elevation', 'Slope']].copy()   # copy, don't change original data
X = X.sample(n=300, random_state=42)    # the same random method
X = X.values                            # convert to numpy array


n_clusters = 12
k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=25, random_state=42)        #def 10 
k_means.fit(X)
k_means_labels = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_

colors_basic = ['#4EACC5', '#FF9C34', '#4E9A06']              #4EACC5 (blue)      FF9C34 (orange)     4E9A06 (green)

colors_all = [
    '#4EACC5', '#FF9C34', '#4E9A06', '#C70039', '#900C3F', '#581845',
    '#FFC300', '#DAF7A6', '#FF5733', '#33FFBD', '#8D33FF', '#FF33A8',
    '#2E86C1', '#28B463', '#A569BD', '#D35400', '#E74C3C', '#1ABC9C',
    '#F1C40F', '#95A5A6', '#34495E', '#E67E22', '#16A085', '#7D3C98',
    '#C0392B', '#5D6D7E', '#48C9B0', '#DC7633', '#99A3A4', '#F39C12'
]



import random
colors = [''] * n_clusters
for i in range(n_clusters):
    colors[i] = random.choice(colors_all)  


plt.figure(figsize=(12,12)) # def 8,6 inches

for k, col in zip(range(n_clusters), colors):   
    my_members = k_means_labels == k                                                                                        # k_means_labels == k: Creates a Boolean mask (True for points belonging to cluster k, False otherwise).
    cluster_center = k_means_cluster_centers[k]
    plt.scatter(X[my_members, 0], X[my_members, 1], color=col, label=f'Cluster {k}', alpha=0.6, edgecolors='k')             # elevation , slope, color, legend = clusters with color description, alpha=0.6: Makes points slightly transparent, edgecolors='k': Draws a black outline around points.
    plt.scatter(cluster_center[0], cluster_center[1], color=col, edgecolors='k', marker='o', s=200, label=f'Center {k}')    # s = 130? (marker size)

plt.title('KMeans Clustering on Elevation & Slope')
plt.xlabel('Elevation')
plt.ylabel('Slope')
plt.legend()
plt.grid(True)
plt.show()
