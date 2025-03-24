
from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from funkcje import *

features = [
    "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points"
]

'''
def clusters_from_centers(zmienna1, zmienna2,calosc, typy, opcja):
    centers = []
    for i in typy:
        a = zmienna1
        b = zmienna2
        tablica_jednego_typu = [calosc[i]]

        if opcja == 1:
            centers.append(opcja_1(tablica_jednego_typu, a, b))
        elif opcja == 2:
            centers.append(opcja_2(tablica_jednego_typu, a, b))
        elif opcja == 3:
            centers.append(opcja_3(tablica_jednego_typu, a, b))
        
    km = KMeans(n_clusters=7, init=centers, max_iter=1)

    covertype = fetch_ucirepo(id=31) 
    y = covertype.data.targets 
    X = covertype.data.features[[a,b]]
    X_sampled = X.sample(n=300, random_state=42).copy()


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sampled)
    X_sampled['cluster'] = km.fit_predict(X_scaled) 
    X_sampled['cluster'] = X_sampled['cluster'].astype('category')

    fig = px.scatter(X_sampled,
                        x=a,
                        y=b,
                        color='cluster')

    fig.show(renderer='browser')        
'''


# def percent_classification(a, b, opcja):
#     centers = []
    
#     # Fetch dataset
#     X_sampled = pd.read_csv("dataset.csv", usecols=[a, b, 'Cover_Type'])
#     X_sampled.dropna()
#     #y = df.columns
#     #X_sampled = df[a, b, 'Cover_Type']
#     #print(X_sampled[:10])
# #percent_classification("Elevation", "Hillshade_Noon", opcja_1)

#     #print(X_sampled['Cover_Type'])


#     #X_sampled = X.sample(n=n, random_state=42).copy() I have already used chosen dataset, so there is no need

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_sampled[[a, b]])

#     #X_sampled['Cover_Type'] = y

#     typy = [1, 2, 3, 4, 5, 6, 7]


#     for i in typy:
#         tablica_jednego_typu = X_sampled[X_sampled['Cover_Type'] == i].copy()

#         if opcja == 1:
#             centers.append(opcja_1(tablica_jednego_typu, a, b))
#         elif opcja == 2:
#             centers.append(opcja_2(tablica_jednego_typu, a, b))
#         elif opcja == 3:
#             centers.append(opcja_3(tablica_jednego_typu, a, b))
#             #print(centers)
    

#     km = KMeans(n_clusters=7, init=centers, max_iter=1)
    
#     X_sampled['cluster'] = km.fit_predict(X_scaled)

#     # Convert to integers to avoid category fuckups
#     X_sampled['cluster'] = X_sampled['cluster'].astype(int)
#     accuracy = (X_sampled['cluster'] == X_sampled['Cover_Type']).mean()
#     return accuracy





def percent_classification(a, b, opcja, iter, manaual_centers_yes):
    # Fetch and sample data
    X_sampled = pd.read_csv("dataset.csv", usecols=[a, b, 'Cover_Type'])

    # Normalize features
    X_scaled = StandardScaler().fit_transform(X_sampled[[a, b]])

    # Choose center calculation function
    opcja_map = {1: opcja_1, 2: opcja_2, 3: opcja_3}
    center_func = opcja_map.get(opcja)
    
    if not center_func:
        raise ValueError("Invalid opcja, must be 1, 2, or 3.")

    if manaual_centers_yes:
        # Calculate centers for each class
        centers = [
            center_func(X_sampled[X_sampled['Cover_Type'] == i], a, b)
            for i in range(1, 8)
        ]
        # Run KMeans with given centers
        km = KMeans(n_clusters=7, init=centers)

    else:
        # Run KMeans without manual centers
        km = KMeans(n_clusters=7, max_iter=iter, n_init=10)


    X_sampled['cluster'] = km.fit_predict(X_scaled)

    # Compute "accuracy" (naive, since KMeans cluster IDs are arbitrary)   
    return (X_sampled['cluster'] == X_sampled['Cover_Type']).mean()



def search_for_5_best(features, iter,opcja, manaual_centers_yes):
    ans = []  # List to store best feature pairs and their accuracy
    
    for i in range(len(features)):
        a = features[i]
        
        for j in range(i + 1, len(features)):  # Avoid (a, a) pairs
            b = features[j]
            
            accuracy = percent_classification(a, b, opcja,iter, manaual_centers_yes)  # Calculate accuracy for feature pair

            # Store [a, b] as an array inside the result list
            ans.append([[a, b], accuracy])
            ans = sorted(ans, key=lambda x: x[1], reverse=True)[:5]  # Keep only top 5

    return ans  # Returns top 5 feature pairs with highest accuracy





def wykresy(a, b, n):
    covertype = fetch_ucirepo(id=31) 
    y = covertype.data.targets 
    X = covertype.data.features[[a,b]]
    X_sampled = X.sample(n=n, random_state=42).copy()


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sampled)

    km = KMeans(n_clusters = 7, random_state=42)

    X_sampled['cluster'] = km.fit_predict(X_scaled) 
    X_sampled['cluster'] = X_sampled['cluster'].astype('category')



    fig = px.scatter(X_sampled,
                        x=a,
                        y=b,
                        color='cluster')
    fig.show(renderer='browser')



    y['Cover_Type'] = y['Cover_Type'].astype('category')
    X_sampled.update(y['Cover_Type'])

    fig2 = px.scatter(X_sampled,
                        x=a,
                        y=b,
                        color='Cover_Type')
    fig2.show(renderer='browser')



