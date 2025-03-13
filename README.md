# forest_clusterring
 project1 - MAD 202 SGGW



pip install ucimlrepo

https://archive.ics.uci.edu/dataset/31/covertype 


Topographical Clustering
Use features: Elevation, Slope, Aspect, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology
Purpose: Identify clusters based on the physical landscape, grouping regions with similar terrain characteristics.
2️⃣ Soil & Geology-Based Clustering
Use features: Soil_Type1 to Soil_Type40, Hillshade_9am, Hillshade_Noon, Hillshade_3pm
Purpose: Find patterns in soil composition and its impact on forest cover.
3️⃣ Hydrology-Based Clustering
Use features: Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology
Purpose: Identify areas with similar proximity to water sources, which may influence vegetation.
4️⃣ Accessibility & Distance Clustering
Use features: Horizontal_Distance_To_Roadways, Horizontal_Distance_To_Fire_Points
Purpose: Group areas based on human accessibility and infrastructure impact.
5️⃣ Vegetation & Cover Type Clustering (Unsupervised Cover Type Prediction)
Use all features (except Cover_Type, since it's the label)
Purpose: Discover natural clusters that align with forest cover types, potentially validating or finding anomalies.


+ zrobić excel z danych ( plikow jest duzo, chodzi by zespolic w jeden, mozna pobrac ze strony)
+ wybrac temat z 5 
+ boxpoloty 5-10 zmiennych
+ klasyfikacja zmiennych
+ 2-3 metody klastrów
+ klasyfikacja z klastrow wzgledem faktow ?
+ do 8 kwietnia