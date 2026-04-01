import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000   # общее количество объектов

# Параметры для двух кластеров (средние и ковариации)
mean1 = [50, 25, 120, 5.0]   # возраст, ИМТ, давление, холестерин
cov1 = [[5, 0, 0, 0], [0, 2, 0, 0], [0, 0, 10, 0], [0, 0, 0, 0.5]]

mean2 = [45, 28, 130, 6.0]
cov2 = [[6, 0, 0, 0], [0, 3, 0, 0], [0, 0, 12, 0], [0, 0, 0, 0.6]]

# Генерируем по 500 точек для каждого кластера
cluster0 = np.random.multivariate_normal(mean1, cov1, 500)
cluster1 = np.random.multivariate_normal(mean2, cov2, 500)

# Создаём перекрытие: меняем 10% точек между кластерами
n_swap = 50
indices0 = np.random.choice(500, n_swap, replace=False)
indices1 = np.random.choice(500, n_swap, replace=False)
cluster0_swapped = cluster0[indices0].copy()
cluster1_swapped = cluster1[indices1].copy()
cluster0[indices0] = cluster1_swapped
cluster1[indices1] = cluster0_swapped

# Объединяем
X = np.vstack([cluster0, cluster1])
y = np.hstack([np.zeros(500), np.ones(500)])

# Добавляем немного шума (нормальное распределение)
noise = np.random.normal(0, 0.5, X.shape)
X += noise

# Преобразуем в DataFrame с реалистичными округлениями
df = pd.DataFrame(X, columns=['age', 'bmi', 'blood_pressure', 'cholesterol'])
df['age'] = df['age'].round(0).astype(int)
df['bmi'] = df['bmi'].round(1)
df['blood_pressure'] = df['blood_pressure'].round(0).astype(int)
df['cholesterol'] = df['cholesterol'].round(2)
df['group'] = y.astype(int)   # истинная принадлежность (для анализа)

# Перемешиваем строки
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Сохраняем
df.to_csv('clustering_data.csv', index=False)
print("Файл clustering_data.csv создан")
print("Размер:", df.shape)
print("\nПервые 5 строк:")
print(df.head())
print("\nБаланс классов:")
print(df['group'].value_counts(normalize=True))