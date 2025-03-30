
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


from sklearn.cluster import Birch
def percent_classification_brc(*args):
    if len(args) < 2:
        raise ValueError("Musisz podać co najmniej 2 kolumny do klastrowania.")

    columns = list(args) + ['Cover_Type']
    X_sampled = pd.read_csv("dataset.csv", usecols=columns)


    X_scaled = StandardScaler().fit_transform(X_sampled[list(args)])

    brc = Birch(n_clusters=7)
    X_sampled['cluster'] = brc.fit_predict(X_scaled)   

    return (X_sampled['cluster'] == X_sampled['Cover_Type']).mean()



def percent_classification2(a, b, opcja, n):
    centers = []
    
    # Fetch dataset
    covertype = fetch_ucirepo(id=31)  
    y = covertype.data.targets
    X = covertype.data.features[[a, b]]
    X_sampled = X.sample(n=n, random_state=42).copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sampled)

    X_sampled['Cover_Type'] = y.loc[X_sampled.index, 'Cover_Type'].astype(int)
    typy = [1, 2, 3, 4, 5, 6, 7]


    for i in typy:
        tablica_jednego_typu = X_sampled[X_sampled['Cover_Type'] == i].copy()

        if opcja == 1:
            centers.append(opcja_1(tablica_jednego_typu, a, b))
        elif opcja == 2:
            centers.append(opcja_2(tablica_jednego_typu, a, b))
        elif opcja == 3:
            centers.append(opcja_3(tablica_jednego_typu, a, b))
            #print(centers)
    

    km = KMeans(n_clusters=7, init=centers, max_iter=15)
    X_sampled['cluster'] = km.fit_predict(X_scaled)

    # Convert to integers to avoid category fuckups
    X_sampled['cluster'] = X_sampled['cluster'].astype(int)
    accuracy = (X_sampled['cluster'] == X_sampled['Cover_Type']).mean()
    return accuracy



def search_for_5_best_brc(features, n):
    ans = [] # List to store best feature pairs and their accuracy

    for i in range(len(features)):
        a = features[i]

        for j in range(i + 1, len(features)): # Avoid (a, a) pairs
            b = features[j]
            if n > 2:
                for k in range(j + 1, len(features)):
                    c = features[k]
                    accuracy = percent_classification_brc(a, b, c)
                    ans.append((a, b, c, accuracy))

            else:
                acc = percent_classification_brc(a, b)
                ans.append([[a, b], acc])

    ans = sorted(ans, key=lambda x: x[1], reverse=True)[:5] # Keep only top 5
    return ans # Returns top 5 feature pairs with highest accuracy







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
        # Run KMeans with automatic centers
        km = KMeans(n_clusters=7, max_iter=iter, n_init=10)


    X_sampled['cluster'] = km.fit_predict(X_scaled)
    return (X_sampled['cluster'] == X_sampled['Cover_Type']).mean()



def percent_classification3(a, b, opcja, n):
    centers = []
    
    # Fetch dataset
    covertype = fetch_ucirepo(id=31)  
    y = covertype.data.targets
    X = covertype.data.features[[a, b]]
    X_sampled = X.sample(n=n, random_state=42).copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sampled)

    X_sampled['Cover_Type'] = y.loc[X_sampled.index, 'Cover_Type'].astype(int)
    typy = [1, 2, 3, 4, 5, 6, 7]


    for i in typy:
        tablica_jednego_typu = X_sampled[X_sampled['Cover_Type'] == i].copy()

        if opcja == 1:
            centers.append(opcja_1(tablica_jednego_typu, a, b))
        elif opcja == 2:
            centers.append(opcja_2(tablica_jednego_typu, a, b))
        elif opcja == 3:
            centers.append(opcja_3(tablica_jednego_typu, a, b))
            #print(centers)
    

    km = KMeans(n_clusters=7, max_iter=15)
    X_sampled['cluster'] = km.fit_predict(X_scaled)

    # Convert to integers to avoid category fuckups
    X_sampled['cluster'] = X_sampled['cluster'].astype(int)
    accuracy = (X_sampled['cluster'] == X_sampled['Cover_Type']).mean()
    return accuracy



def search_for_5_best(features):
    ans = [] # List to store best feature pairs and their accuracy

    for i in range(len(features)):
        a = features[i]

        for j in range(i + 1, len(features)): # Avoid (a, a) pairs
            b = features[j]

            for iter in range(1,10):

                for opcja in range(1,3):
                    accuracy = percent_classification(a, b, opcja,iter, True)
                    ans.append([[a, b], accuracy])


                acc = percent_classification(a, b, 1,iter, False)
                ans.append([[a, b], acc])   



    ans = sorted(ans, key=lambda x: x[1], reverse=True)[:5] # Keep only top 5
    return ans # Returns top 5 feature pairs with highest accuracy







from sklearn.cluster import DBSCAN

def percent_classification_dbscan(a, b):
    # Fetch and sample data
    X_sampled = pd.read_csv("dataset.csv", usecols=[a, b, 'Cover_Type'])

    # Normalize features
    X_scaled = StandardScaler().fit_transform(X_sampled[[a, b]])

    dbscan = DBSCAN(eps=1.5, min_samples=7)  # Adjust parameters as needed

    X_sampled['cluster'] = dbscan.fit_predict(X_scaled)

    # Compute "accuracy" (naive, since KMeans cluster IDs are arbitrary)   
    return (X_sampled['cluster'] == X_sampled['Cover_Type']).mean()



def search_for_5_best_dbscan(features):
    ans = [] # List to store best feature pairs and their accuracy

    for i in range(len(features)):
        a = features[i]

        for j in range(i + 1, len(features)): # Avoid (a, a) pairs
            b = features[j]
            acc = percent_classification_dbscan(a, b)
            ans.append([[a, b], acc])   



    ans = sorted(ans, key=lambda x: x[1], reverse=True)[:5] # Keep only top 5
    return ans # Returns top 5 feature pairs with highest accuracy





def search_for_5_best_oldversion(features, iter,opcja, manaual_centers_yes):
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


















'''    colors_basic = ['#4EACC5', '#FF9C34', '#4E9A06']              #4EACC5 (blue)      FF9C34 (orange)     4E9A06 (green)

    colors_all = [
        '#4EACC5', '#FF9C34', '#4E9A06', '#C70039', '#900C3F', '#581845',
        '#FFC300', '#DAF7A6', '#FF5733', '#33FFBD', '#8D33FF', '#FF33A8',
        '#2E86C1', '#28B463', '#A569BD', '#D35400', '#E74C3C', '#1ABC9C',
        '#F1C40F', '#95A5A6', '#34495E', '#E67E22', '#16A085', '#7D3C98',
        '#C0392B', '#5D6D7E', '#48C9B0', '#DC7633', '#99A3A4', '#F39C12']
    
    import random
    colors = [''] * n_clusters
    for i in range(n_clusters):
        colors[i] = random.choice(colors_all)  
        '''




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