# 3D old - might be wrong a bit, clusters look odd

from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
import matplotlib.pyplot as plt

  
# fetch dataset 
covertype = fetch_ucirepo(id=31) 
  
# data (as pandas dataframes) 
X = covertype.data.features 
X2 = X.sample(n=300, random_state=42).copy()
y = covertype.data.targets 
  

km = KMeans(n_clusters = 10, random_state=42)

X2['cluster'] = km.fit_predict(X2) 
X2['cluster'] = X2['cluster'].astype('category')



fig = px.scatter_3d(X2,
                    x='Elevation',
                    y='Aspect',
                    z='Slope',
                    color='cluster')

fig.show(renderer='browser')








