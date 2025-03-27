
from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from funkcje import *
from main_functions import *


# each time we run the script, we fetch repo, so maybe it's faster to save it in a file and use that way

centers = []

covertype = fetch_ucirepo(id=31)

y = covertype.data.targets 
a = 'Elevation'
b = 'Aspect'
X = covertype.data.features[[a,b]]
X_sampled = X.sample(n=200, random_state=42).copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sampled)
y['Cover_Type'] = y['Cover_Type'].astype('category')
X_sampled['Cover_Type'] = y['Cover_Type']

'''
print(X_sampled['Cover_Type'])
print(percent_classification('Elevation', 'Slope', 3, 10, False))
print('\n')

print(percent_classification('Elevation', 'Slope', 3, 10, True))
print('\n')

print(percent_classification('Elevation', 'Slope', 1, 10, False))
print('\n')

print(percent_classification('Elevation', 'Slope', 1, 10, True))

print(percent_classification('Elevation', 'Slope', 1, 5, True))
print('\n')

print(percent_classification('Elevation', 'Slope', 1, 5, False))
print('\n')

print(percent_classification('Elevation', 'Slope', 3, 5, True))  #0.238
print('\n')

print(percent_classification('Elevation', 'Slope', 3, 5, False))
'''


features = [
    "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points"
]
best = search_for_5_best_brc(features=features) 

#print(best)
for i in best:
    print(i[0], i[1])