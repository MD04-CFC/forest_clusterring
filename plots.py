
from ucimlrepo import fetch_ucirepo 
from sklearn.cluster import KMeans
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from main_functions import *
  


def wykres_1(a, b, opcja):
    opcja_map = {1: opcja_1, 2: opcja_2, 3: opcja_3}
    center_func = opcja_map.get(opcja)
    

    X_sampled = pd.read_csv("dataset.csv", usecols=[a, b, 'Cover_Type'])


    X_scaled = StandardScaler().fit_transform(X_sampled[[a, b]])

    centers = [
                center_func(X_sampled[X_sampled['Cover_Type'] == i], a,b)
                for i in range(1, 8)
            ]
    km = KMeans(n_clusters=7, init=centers)


    X_sampled['cluster'] = km.fit_predict(X_scaled) 
    X_sampled['cluster'] = X_sampled['cluster'].astype('category')



    fig = px.scatter(X_sampled,
                        x='Elevation',
                        y='Slope',
                        color='cluster',
                        title='Clusters from KMeans')

    fig.show()




def wykres_2(a, b, opcja):
    opcja_map = {1: opcja_1, 2: opcja_2, 3: opcja_3}
    center_func = opcja_map.get(opcja)
    X_sampled = pd.read_csv("dataset.csv", usecols=[a, b, 'Cover_Type'])
    #X = StandardScaler().fit_transform(X_sampled[[a, b]])
    X = X_sampled[[a, b]].values

    centers = [
                center_func(X_sampled[X_sampled['Cover_Type'] == i], a,b)
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

        plt.scatter(X[my_members, 0], X[my_members, 1], color=col, label=f'Cluster {k+1}', alpha=0.6, edgecolors='k')                                                                     # elevation , slope, color, legend = clusters with color description, alpha=0.6: Makes points slightly transparent, edgecolors='k': Draws a black outline around points.
        plt.scatter(cluster_center[0], cluster_center[1], color=col, edgecolors='k', marker='o', s=200, label=f'Center {k+1}')                                                            # s = 130? (marker size)

    plt.title(f'KMeans Clustering on {a} & {b}')
    plt.xlabel(a)
    plt.ylabel(b)
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
    for c in range(7):
        my_members = covertype_labels == c
        #plt.scatter(X[my_members, 0], X[my_members, 1], color = fixed_colors[c], label=f'Covertype {c}', alpha=0.6, edgecolors='k')
        plt.scatter(X_sampled[a][my_members], X_sampled[b][my_members], color=fixed_colors[c], label=f'Covertype {c+1}', alpha=0.6, edgecolors='k')
    
    plt.xlabel(a)
    plt.ylabel(b)
    plt.title(f'Plot of {a} and {b} with covertype')
    plt.legend(title='Covertype')
    plt.grid(True)
    plt.show()
    


wykres_2('Hillshade_3pm', 'Hillshade_9am', 1)






















