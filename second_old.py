# 2D before AI , it's ok


from ucimlrepo import fetch_ucirepo 
from sklearn.cluster import KMeans
import plotly.express as px
from sklearn.preprocessing import StandardScaler
  
# fetch dataset 
covertype = fetch_ucirepo(id=31) 
  
# data (as pandas dataframes) 
y = covertype.data.targets 
X = covertype.data.features[['Elevation','Slope']]
X_sampled = X.sample(n=300, random_state=42).copy()

#print(y.head())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sampled)

km = KMeans(n_clusters = 7, random_state=42)

X_sampled['cluster'] = km.fit_predict(X_scaled) 
X_sampled['cluster'] = X_sampled['cluster'].astype('category')



fig = px.scatter(X_sampled,
                    x='Elevation',
                    y='Slope',
                    color='cluster',
                    title='Clusters from KMeans')

fig.show()
