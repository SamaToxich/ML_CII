import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv('clustering_data.csv')

print("\nПервые 5 строк:")
print(df.head())

X = df.drop('group', axis=1)
y = df['group']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=2)
clusters = kmeans.fit_predict(X_scaled)

ct = pd.crosstab(y, clusters)
print("\nТаблица сопряжения (диагноз vs кластер):")
print(ct)