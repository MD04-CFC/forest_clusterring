from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
covertype = fetch_ucirepo(id=31) 
  
# data (as pandas dataframes) 
X = covertype.data.features 
X2 = X.copy()
y = covertype.data.targets 
  
print(X.head())


import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px

km = KMeans(n_clusters = 6)

    
X2['cluster'] = km.fit_predict(X) 
X2['cluster'] = X2['cluster'].astype('category')



fig = px.scatter_3d(X2,
                    x='Elevation',
                    y='Aspect',
                    z='Slope',
                    color='cluster')

fig.show(renderer='browser')'
