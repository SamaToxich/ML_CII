import math
import random
import time
import numpy as np
from matplotlib import pyplot as plt
from numpy.random import rand
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score


def euclidean_distance(p1,p2):
    """Евклидово расстояние между двумя точками в n-мерном пространстве."""
    return math.sqrt(sum((a - b) ** 2 for a,b in zip(p1, p2)))

def cdist(X, Y=None):
    """Вычисляет матрицу попарных расстояний.
    Возвращает список списков (матрицу) float."""
    if Y is None:
        Y = X

    n = len(X)
    m = len(Y)

    dist_matrix = [[0.0] * m for _ in range(n)]

    for i in range(n):
        for j in range(m):
            dist_matrix[i][j] = euclidean_distance(X[i], Y[j])
    return dist_matrix

def prim_mst(dist_matrix):
    """dist_matrix — квадратная матрица попарных расстояний (list of lists).
    Возвращает список рёбер MST: каждое ребро — (i, j, вес)."""

    n = len(dist_matrix)
    if n <= 1: return []

    visited = [False] * n
    visited[0] = True
    edges = []

    for _ in range(n - 1):
        # Ищем минимальное ребро из посещённых вершин в не посещённые
        min_weight = float('inf') # бесконечно большое число
        u, v = -1, -1

        for i in range(n):
            if visited[i]:
                for j in range(n):
                    if not visited[j] and dist_matrix[i][j] < min_weight:
                        min_weight = dist_matrix[i][j]
                        u, v = i, j

        if u == -1:
            break

        visited[v] = True
        edges.append((u, v, min_weight))

    return edges

def kruskal_mst(points):
    n = len(points)
    if n <= 1: return []

    # Матрица расстояний через cdist
    dist_mat = cdist(points)   # квадратная матрица n x n

    # Генерируем все рёбра (i, j, вес)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            w = euclidean_distance(points[i], points[j])
            edges.append((i, j, w))

    # Сортируем по весу
    edges.sort(key=lambda x: x[2])

    # DSU (система непересекающихся множеств)
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        if parent[x] == x:
            return x
        parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        x, y, = find(x), find(y)
        if x == y:
            return False
        if random.randint(1, 10) % 2 == 0:
            x, y = y, x
        parent[x] = y
        return True

    mst_edges = []

    for i, j, w in edges:
        if union(i,j):
            mst_edges.append((i,j,w))
            if len(mst_edges) == n - 1:
                break
    return mst_edges

# ------------------------------------------------------------
# 1. Алгоритм кластеризации на основе минимального остовного дерева (MST)
# ------------------------------------------------------------
def mst_clustering(points, k):
    n = len(points)
    if k >= n: return list(range(n))

    # Построение минимального остовного дерева
    mst_edges = kruskal_mst(points) # уже отсортированы по возрастанию

    # DSU для объединения компонент от коротких рёбер
    parent = list(range(n))
    rank = [0] * n

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    def union(v1, v2):
        r1, r2, = find(v1), find(v2)
        if r1 == r2:
            return False
        if rank[r1] < rank[r2]:
            parent[r1] = r2
        elif rank[1] > rank[2]:
            parent[r2] = r1
        else:
            parent[r2] = r1
            rank[r1] += 1
        return True

    components = n
    for i, j, w in mst_edges:
        if components == k:
            break
        if union(i, j):
            components -= 1

    labels = np.empty(n, dtype=int)

    # Собираем корни для каждой точки
    roots = [find(i) for i in range(n)]

    # Уникальные корни (их должно быть ровно k)
    unique_roots = {}

    for r in roots:
        if r not in unique_roots:
            unique_roots[r] = len(unique_roots)

    for i,r in enumerate(roots):
        labels[i] = unique_roots[r]
    return labels

# ------------------------------------------------------------
# 2. Реализация K-means с нуля
# ------------------------------------------------------------
def my_kmeans(X, k, max_iters=100, tol=1e-4, random_state=None):
    """K-means с инициализацией случайными точками из набора.
    Возвращает метки кластеров и центроиды."""

    if random_state is not None:
        np.random.seed(random_state)

    row, col = X.shape # Узнаём сколько строк и столбцов

    # Случайный выбор k точек в качестве начальных центроидов
    indices = np.random.choice(row, k, replace=False)
    centroids = X[indices].copy()

    for _ in range(max_iters):
        # Расстояния до центроидов
        distances = cdist(X, centroids)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.zeros((k, col))

        for j in range(k):
            if np.sum(labels == j) > 0:
                new_centroids[j] = X[labels == j].mean(axis=0)
            else:
                # Если кластер пуст, оставляем старый центроид
                new_centroids[j] = centroids[j]

        # Проверка сходимости
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids

    return labels, centroids

# ------------------------------------------------------------
# 3. Функция для оценки качества
# ------------------------------------------------------------
def evaluate_clustering(X, labels, centroids=None):
    """
    Возвращает силуэт и инерцию (сумму квадратов расстояний до центроидов).
    Если centroids не заданы, вычисляются как средние по кластерам.
    """
    if len(set(labels)) < 2:
        # Силуэт требует как минимум 2 кластера
        sil = -1.0
    else:
        sil = silhouette_score(X, labels, metric='euclidean')

    if centroids is None:
        # Вычисляем центроиды по меткам
        k = len(np.unique(labels))
        centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])

    inertia = 0.0
    for j, c in enumerate(centroids):
        inertia += np.sum((X[labels == j] - c) ** 2)
    return sil, inertia

# ------------------------------------------------------------
# 4. Генерация тестовых наборов
# ------------------------------------------------------------
def generate_test_configs():
    """
    Возвращает список конфигураций для тестирования.
    Каждая конфигурация: (N, K, описание, функция-генератор точек)
    """
    configs = []
    # 1. Три компактных блоба, N=150, K=3
    configs.append(('blobs_3_150', 150, 3, lambda: make_blobs(n_samples=150, centers=3, cluster_std=0.8, random_state=42)[0]))
    # 2. Пять блобов, N=300, K=5
    configs.append(('blobs_5_300', 300, 5, lambda: make_blobs(n_samples=300, centers=5, cluster_std=1.0, random_state=42)[0]))
    # 3. Два блоба, N=80, K=2
    configs.append(('blobs_2_80', 80, 2, lambda: make_blobs(n_samples=80, centers=2, cluster_std=1.2, random_state=42)[0]))
    # 4. Восемь блобов, N=400, K=8
    configs.append(('blobs_8_400', 400, 8, lambda: make_blobs(n_samples=400, centers=8, cluster_std=0.9, random_state=42)[0]))
    # 5. Десять блобов, N=500, K=10
    configs.append(('blobs_10_500', 500, 10, lambda: make_blobs(n_samples=500, centers=10, cluster_std=1.1, random_state=42)[0]))
    # 6. Равномерное распределение (нет явных кластеров), N=200, K=4
    configs.append(('uniform_200_4', 200, 4, lambda: np.random.RandomState(42).uniform(-5, 5, size=(200, 2))))
    # 7. Разреженные точки (два удалённых скопления + шум), N=120, K=3
    def sparse_blobs():
        np.random.seed(42)
        centers = np.array([[0,0], [10,10], [20,0]])
        points = []
        for c in centers:
            pts = c + np.random.randn(40, 2) * 0.5
            points.append(pts)
        return np.vstack(points)
    configs.append(('sparse_3_120', 120, 3, sparse_blobs))

    return configs

# ------------------------------------------------------------
# 5. Проведение эксперимента
# ------------------------------------------------------------
def run_experiment():
    configs = generate_test_configs()
    results = []

    print("=== Сравнение алгоритмов кластеризации ===\n")
    print(f"{'Тест':<20} {'N':<5} {'K':<3} {'Метод':<12} {'Время (с)':<10} {'Силуэт':<8} {'Инерция':<12}")
    print("-" * 80)

    for name, N, K, gen_func in configs:
        X = gen_func()
        # Убедимся, что K не больше N
        K = min(K, N)

        # --- MST кластеризация ---
        start = time.perf_counter()
        labels_mst = mst_clustering(X, K)
        time_mst = time.perf_counter() - start
        # Вычисляем центроиды для инерции (по средним)
        centroids_mst = np.array([X[labels_mst == j].mean(axis=0) for j in range(K)])
        sil_mst, inertia_mst = evaluate_clustering(X, labels_mst, centroids_mst)

        # --- Моя реализация K-means ---
        start = time.perf_counter()
        labels_my, centroids_my = my_kmeans(X, K, random_state=42)
        time_my = time.perf_counter() - start
        sil_my, inertia_my = evaluate_clustering(X, labels_my, centroids_my)

        # --- K-means из scikit-learn ---
        start = time.perf_counter()
        kmeans_sk = KMeans(n_clusters=K, random_state=42, n_init=10)
        labels_sk = kmeans_sk.fit_predict(X)
        time_sk = time.perf_counter() - start
        sil_sk, inertia_sk = evaluate_clustering(X, labels_sk, kmeans_sk.cluster_centers_)

        # Сохраняем результаты
        results.append({
            'name': name, 'N': N, 'K': K,
            'mst': (time_mst, sil_mst, inertia_mst),
            'my_kmeans': (time_my, sil_my, inertia_my),
            'sklearn_kmeans': (time_sk, sil_sk, inertia_sk)
        })

        # Вывод строки для каждого метода
        print(f"{name:<20} {N:<5} {K:<3} {'MST':<12} {time_mst:<10.5f} {sil_mst:<8.4f} {inertia_mst:<12.2f}")
        print(f"{name:<20} {N:<5} {K:<3} {'MyKMeans':<12} {time_my:<10.5f} {sil_my:<8.4f} {inertia_my:<12.2f}")
        print(f"{name:<20} {N:<5} {K:<3} {'Sklearn':<12} {time_sk:<10.5f} {sil_sk:<8.4f} {inertia_sk:<12.2f}")
        print("-" * 80)

    return results

# ------------------------------------------------------------
# Запуск эксперимента и визуализации
# ------------------------------------------------------------
if __name__ == "__main__":
    results = run_experiment()
    visualize_example()