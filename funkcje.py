import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv")

features = [
    "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points"
]

X = df[features]


def wykres2D_scatter(a,b):             
    plt.figure(figsize=(7, 5))
    plt.scatter(df[f'{a}'], df[f'{b}'], color='purple', alpha=0.6)
    plt.title(f'{a} vs {b}')
    plt.xlabel(f'{a}')
    plt.ylabel(f'{b}')
    plt.grid(True)
    plt.show()

def wykres2D_plot(a, b):
    plt.figure(figsize=(8, 5))
    plt.plot(df[f'{a}'], df[f'{b}'], marker='o', linestyle='-', color='green')
    plt.title(f'{a} to {b}')
    plt.xlabel(f'{a}')
    plt.ylabel(f'{b}')
    plt.grid(True)
    plt.show()

def wykres3D(a,b, c):             
    import plotly.express as px


def histogram(a):  
    plt.figure(figsize=(8, 5))
    plt.hist(df[f'{a}'], bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {a}')
    plt.xlabel(f'{a}')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()     


def boxplot(a):   
    plt.figure(figsize=(6, 5))
    plt.boxplot(df[f'{a}'].dropna(), vert=True)
    plt.title(f'Boxplot of {a}')
    plt.ylabel(f'{a}')
    plt.grid(True)
    plt.show()  


def calculate_basic_statistical_measures(a):
    mean = df[f"{a}"].mean()
    median = df[f"{a}"].median()
    mode = df[f"{a}"].mode().iloc[0]

    # Dispersion
    std_dev = df[f"{a}"].std()
    variance = df[f"{a}"].var()
    range_val = df[f"{a}"].max() - df[f"{a}"].min()
    iqr = df[f"{a}"].quantile(0.75) - df[f"{a}"].quantile(0.25)

    # Position
    min_val = df[f"{a}"].min()
    max_val = df[f"{a}"].max()

    print(f"Mean: {mean}, Median: {median}, Mode: {mode}, Variance: {variance}")
    print(f"Minimalna wartosc: {min_val}, maxymalna wartosc: {max_val}")
    print(f"Standard Deviation: {std_dev}, IQR: {iqr}, Range: {range_val}")


def srednia(tablica):
    return sum(tablica)/len(tablica)

def opcja_1(tablica_jednego_typu, a, b):                                                          #_srodki_srednia
    return [tablica_jednego_typu[a].mean(), tablica_jednego_typu[a].mean() ]               




def wspolrzedne_punktu_z_danego_typu_2D(x, tablica_jednego_typu):
    return 1
    # return wyborpunktu[x]



def opcja_2(tablica_jednego_typu, a,b):                                                            #_srodki_z_danego_typu   --> wybor                   
    return [ wspolrzedne_punktu_z_danego_typu_2D(a, tablica_jednego_typu), wspolrzedne_punktu_z_danego_typu_2D(b, tablica_jednego_typu) ]         




import random
def opcja_3(tablica_jednego_typu,a,b):                      #_srodki_losowe
    x = tablica_jednego_typu.sample(n=1).iloc[0]  

    return [x.iloc[0], x.iloc[1]]  
                                               


def opcja_3_3D(tablica_jednego_typu,a,b,c):          #_srodki_losowe
    x = tablica_jednego_typu.sample(n=1).iloc[0]  

    return [x.iloc[0], x.iloc[1], x.iloc[2]]            



def srednia(tablica):
    return sum(tablica)/len(tablica)

def opcja_1_3D(tablica_jednego_typu, a, b,c):                                                          
    return [ srednia(tablica_jednego_typu[a]), srednia(tablica_jednego_typu[b]), srednia(tablica_jednego_typu[c]) ]         





