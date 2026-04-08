import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv('data/clustering_data.csv')

print("\nПервые 5 строк:")
print(df.head())

X = df.drop('group', axis=1)
y = df['group']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(len(np.unique(y)), random_state=42)
clusters = kmeans.fit_predict(X_scaled)
print(clusters)

ct = pd.crosstab(y, clusters)
print(f"\nТаблица схождения:\n{ct}")
print(f'Точность: {accuracy_score(y, clusters)}')
print(confusion_matrix(y, clusters))
print(f'{classification_report(y, clusters)}')
