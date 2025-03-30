from ucimlrepo import fetch_ucirepo 
from sklearn.cluster import KMeans
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from main_functions import *
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.io as pio



def wykres_3D_html(a, b, c,n, opcja):
    opcja_map = {1: opcja_1_3D, 3: opcja_3_3D}
    center_func = opcja_map.get(opcja)

    X_sampled = pd.read_csv("dataset.csv", usecols=[a, b, c, 'Cover_Type'])
    X_sampled = X_sampled.sample(n=n, random_state=42).copy()
    X = X_sampled[[a, b, c]].values

    centers = [
        center_func(X_sampled[X_sampled['Cover_Type'] == i], a, b, c)
        for i in range(1, 8)
    ]

    n_clusters = 7
    k_means = KMeans(init=centers, n_clusters=n_clusters, n_init=8, random_state=42)
    k_means.fit_predict(X)
    X_sampled['Cluster'] = k_means.labels_+1

    color_map = {
         '1': 'red', '2': 'green', '3': 'purple',
        '4': 'orange', '5': 'cyan', '6': 'brown','7': 'blue'
    }

    # 3D K-Means Clusters
    fig_clusters = px.scatter_3d(X_sampled, x=a, y=b, z=c, 
                                  color=X_sampled['Cluster'].astype(str), 
                                  title="K-Means Clustering",
                                  labels={a: a, b: b, c: c},
                                  color_discrete_map=color_map)
    
    # 3D Cover_Type
    fig_cover = px.scatter_3d(X_sampled, x=a, y=b, z=c, 
                               color=X_sampled['Cover_Type'].astype(str), 
                               title="Cover Type Distribution",
                               labels={a: a, b: b, c: c},
                               color_discrete_map=color_map)
    
    # html
    # Example time string
    import time
    from datetime import datetime
    time_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    time_format = "%Y-%m-%d %H:%M:%S.%f"
    parsed_time = datetime.strptime(time_string, time_format)

    pio.write_html(fig_clusters, file=f"{parsed_time.microsecond}_clusters_3D.html", auto_open=True)
    pio.write_html(fig_cover, file=f"{parsed_time.microsecond}_cover_3D.html", auto_open=True)






def wykres_3D_both(a, b, c,n, opcja=3):
    opcja_map = {1: opcja_1, 2: opcja_2, 3: opcja_3_3D}
    center_func = opcja_map.get(opcja)

    X_sampled = pd.read_csv("dataset.csv", usecols=[a, b, c, 'Cover_Type'])
    X = X_sampled.sample(n=n, random_state=42).copy()
    X_sampled = X 
    X = X[[a, b, c]].values

    centers = [
        center_func(X_sampled[X_sampled['Cover_Type'] == i], a, b, c)
        for i in range(1, 8)
    ]

    n_clusters = 7
    k_means = KMeans(init=centers, n_clusters=n_clusters, n_init=8, random_state=42)
    k_means.fit_predict(X)
    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_

    colors_all = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'brown', 'pink']
    fixed_colors = {i: colors_all[i] for i in range(n_clusters)}

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    for k in range(n_clusters):
        my_members = k_means_labels == k
        cluster_center = k_means_cluster_centers[k]
        col = fixed_colors[k]

        ax.scatter(X[my_members, 0], X[my_members, 1], X[my_members, 2], 
                   color=col, label=f'Cluster {k+1}', alpha=0.6, edgecolors='k')
        ax.scatter(cluster_center[0], cluster_center[1], cluster_center[2], 
                   color=col, edgecolors='k', marker='o', s=200, label=f'Center {k+1}')

    ax.set_xlabel(a)
    ax.set_ylabel(b)
    ax.set_zlabel(c)
    ax.legend()
    plt.grid(True)
    plt.show()

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    covertype_labels = X_sampled['Cover_Type'] - 1

    for d in range(7):
        my_members = covertype_labels == d
        ax.scatter(X_sampled[a][my_members], X_sampled[b][my_members], X_sampled[c][my_members], 
                   color=fixed_colors[d], label=f'Covertype {d+1}', alpha=0.6, edgecolors='k')

    ax.set_xlabel(a)
    ax.set_ylabel(b)
    ax.set_zlabel(c)
    ax.legend(title='Covertype')
    plt.grid(True)
    plt.show()


























def wykres_3D_both2(a, b,c, opcja=3):
    opcja_map = {1: opcja_1, 2: opcja_2, 3: opcja_3_3D}
    center_func = opcja_map.get(opcja)
    X_sampled = pd.read_csv("dataset.csv", usecols=[a, b,c, 'Cover_Type'])
    X = X_sampled[[a, b,c]].values

    centers = [
                center_func(X_sampled[X_sampled['Cover_Type'] == i], a,b,c)
                for i in range(1, 8)
            ]
    
    n_clusters = 7
    k_means = KMeans(init=centers, n_clusters=n_clusters, n_init=8, random_state=42)         
    k_means.fit_predict(X)
    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_



    colors_all = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'brown', 'pink']  
    fixed_colors = {i: colors_all[i] for i in range(n_clusters)}  

    
    plt.figure(figsize=(12,12))         # def 8,6 inches

    for k in range(n_clusters):   
        my_members = k_means_labels == k                                                                                                                                                # k_means_labels == k: Creates a Boolean mask (True for points belonging to cluster k, False otherwise).
        cluster_center = k_means_cluster_centers[k]
        col = fixed_colors[k]

        plt.scatter(X[my_members, 0], X[my_members, 1],X[my_members, 2], color=col, label=f'Cluster {k+1}', alpha=0.6, edgecolors='k')                                                                     # elevation , slope, color, legend = clusters with color description, alpha=0.6: Makes points slightly transparent, edgecolors='k': Draws a black outline around points.
        plt.scatter(cluster_center[0], cluster_center[1],cluster_center[2], color=col, edgecolors='k', marker='o', s=200, label=f'Center {k+1}')                                                            # s = 130? (marker size)


    plt.xlabel(a)
    plt.ylabel(b)
    plt.zlabel(c)
    plt.legend()
    plt.grid(True)
    plt.show(block=False)



    covertype_labels = X_sampled['Cover_Type']-1

    '''
    from sklearn.metrics import adjusted_rand_score
    ari_score = adjusted_rand_score(covertype_labels, k_means_labels)
    print(f"Adjusted Rand Index (ARI): {ari_score}")
    '''
    
    plt.figure(figsize=(12,12))
    for d in range(7):
        my_members = covertype_labels == d
        plt.scatter(X_sampled[a][my_members], X_sampled[b][my_members],X_sampled[c][my_members], color=fixed_colors[d], label=f'Covertype {d+1}', alpha=0.6, edgecolors='k')
    
    plt.xlabel(a)
    plt.ylabel(b)
    plt.zlabel(c)
    plt.legend(title='Covertype')
    plt.grid(True)
    plt.show()
    
