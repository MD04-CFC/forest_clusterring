import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv")
df = df.sample(frac=0.01, random_state=42)

features = [
    "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points"
]

X = df[features]

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



