# 2D before AI , it's ok


from ucimlrepo import fetch_ucirepo
from sklearn.cluster import KMeans
import plotly.express as px
import pandas as pd
from sklearn.preprocessing import StandardScaler
  
# fetch dataset 
covertype = pd.read_csv("dataset.csv") 

# data (as pandas dataframes) 
y = covertype['Cover_Type']
X = covertype[['Elevation','Slope']]
X_sampled = X.sample(n=3000, random_state=42)

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
