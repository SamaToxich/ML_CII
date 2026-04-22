import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv('../data/clustering_data.csv')

X = df.drop('group', axis=1)
y = df['group']

scaler = StandardScaler()
X = scaler.fit_transform(X)

k_range = range(2,100)
indexCX = []
cnt_down = 0

for k in k_range:
    kmeans = KMeans(k, random_state=42)
    labels = kmeans.fit_predict(X)
    cur_CX = silhouette_score(X, labels) / kmeans.inertia_ * ((X.shape[0] - k) / (k-1))
    indexCX.append(cur_CX)

    if cur_CX < max(indexCX):
        cnt_down += 1
        if cnt_down >= 3:
            break

best_k = k_range[np.argmax(indexCX)]

model = KMeans(best_k, random_state=42)
clusters = model.fit_predict(X)

print(f'\nЛучшее K: {best_k}')
print(f'Инерция: {model.inertia_}')
print(f'Силуэт: {silhouette_score(X, clusters)}')