# 2D before AI --  -- something is wrong, clusters are really odd and probably not correct


from ucimlrepo import fetch_ucirepo 
from sklearn.cluster import KMeans
import plotly.express as px
  
# fetch dataset 
covertype = fetch_ucirepo(id=31) 
  
# data (as pandas dataframes) 
X2 = covertype.data.features 
y = covertype.data.targets 
X = X2.sample(n=300, random_state=42).copy()   # the same random method



km = KMeans(n_clusters = 10, random_state=42)

X['cluster'] = km.fit_predict(X) 
X['cluster'] = X['cluster'].astype('category')



fig = px.scatter(X,
                    x='Elevation',
                    y='Slope',
                    color='cluster')

fig.show(renderer='browser')