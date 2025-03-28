
from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from funkcje import *



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



def percent_classification(a, b, opcja, n):
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
    

    km = KMeans(n_clusters=7, init=centers, max_iter=1)
    X_sampled['cluster'] = km.fit_predict(X_scaled)

    # Convert to integers to avoid category fuckups
    X_sampled['cluster'] = X_sampled['cluster'].astype(int)
    accuracy = (X_sampled['cluster'] == X_sampled['Cover_Type']).mean()
    return accuracy



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



def najlepsze(n):
    kombinacje = []         #kombinacje zmiennych do zrobienia!!!!!
    najlepsza = 0
    najlepsza_komb = []     # najlepsza kombinacja zmiennych i opcji

    for i in kombinacje:
        a = kombinacje[0]
        b = kombinacje[1]

        for j in range(3):
            wynik = percent_classification(a,b,j,n)
            if wynik > najlepsza:
                najlepsza = wynik
                najlepsza_komb = [a,b,j]
        
    return najlepsza_komb



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