import numpy as np
import pandas as pd



def find_outliers(data):
 
    if isinstance(data, (list, np.ndarray)):
        data = pd.Series(data)
    
    # quartiles
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    
    # IQR (Interquartile Range)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    

    cleaned_data = data[(data >= lower_bound) & (data <= upper_bound)]
    
    return cleaned_data


features = [
    "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points"
]

def remove_outliers(df, features):
    for column in features:
        cleaned_column = find_outliers(df[column])
        df = df[df[column].isin(cleaned_column)]
    return df


file_path = 'dataset.csv'  #csv
df = pd.read_csv(file_path)
file_path2 = 'dataset_clean.csv'  
cleaned_df = remove_outliers(df, features)
print(cleaned_df.head())
cleaned_df.to_csv(file_path2, index=False)
