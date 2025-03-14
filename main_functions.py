
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




def percent_classification(zmienna1, zmienna2, opcja, n):
    centers = []
    covertype = fetch_ucirepo(id=31) 
    y = covertype.data.targets 
    X = covertype.data.features[[a,b]]
    X_sampled = X.sample(n=n, random_state=42).copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sampled)
    typy = X.columns

    for i in typy:
        a = zmienna1
        b = zmienna2
        tablica_jednego_typu = [X_scaled[i]]

        if opcja == 1:
            centers.append(opcja_1(tablica_jednego_typu, a, b))
        elif opcja == 2:
            centers.append(opcja_2(tablica_jednego_typu, a, b))
        elif opcja == 3:
            centers.append(opcja_3(tablica_jednego_typu, a, b))
        
    km = KMeans(n_clusters=7, init=centers, max_iter=1)

    
    X_sampled['cluster'] = km.fit_predict(X_scaled) 
    X_sampled['cluster'] = X_sampled['cluster'].astype('category')

    ile = 0  
    ile_popr = 0

    for i in range(n):
        if X_sampled['cluster'][i] == y[i]:
            ile_popr += 1           
        ile += 1

    return ile_popr/ile




def najlepsze(n):
    kombinacje = []         #kombinacje zmiennych do zrobienia!!!!!
    najlepsza = 0
    najlepsza_komb = []     # najlepsza kombinacja zmiennych i opcji

    for i in kombinacje:
        a = kombinacje[0]
        b = kombinacje[1]
        covertype = fetch_ucirepo(id=31) 
        y = covertype.data.targets 
        X = covertype.data.features[[a,b]]

        for j in range(3):
            wynik = percent_classification(a,b,j,n)
            if wynik > najlepsza:
                najlepsza = wynik
                najlepsza_komb = [a,b,j]
        
    return najlepsza_komb



def wykres(a, b, n):
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