#3D new - correct clusters

from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Fetch dataset
covertype = fetch_ucirepo(id=31) 


X = covertype.data.features[['Elevation', 'Aspect', 'Slope']]
X_sampled = X.sample(n=300, random_state=42).copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sampled)


n_clusters = 7
km = KMeans(n_clusters=n_clusters, random_state=42)
X_sampled['cluster'] = km.fit_predict(X_scaled)
X_sampled['cluster'] = X_sampled['cluster'].astype('category')

fig = px.scatter_3d(X_sampled,
                    x='Elevation',
                    y='Aspect',
                    z='Slope',
                    color='cluster')

#fig.show(renderer='browser')
fig.write_html("3d_cluster_plot.html")









