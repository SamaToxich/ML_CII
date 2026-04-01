import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# -------------------------------
# 1. Чтение данных из таблицы
# -------------------------------
data = {
    'продукт': ['Яблоко', 'салат', 'бекон', 'банан', 'орехи', 'рыба', 'сыр', 'виноград', 'морковь', 'апельсин'],
    'сладость': [7, 2, 1, 9, 1, 1, 1, 8, 2, 6],
    'хруст': [7, 5, 2, 1, 5, 1, 1, 1, 8, 1],
    'класс': ['Фрукт', 'Овощ', 'Протеин', 'Фрукт', 'Протеин', 'Протеин', 'Протеин', 'Фрукт', 'Овощ', 'Фрукт']
}

df = pd.DataFrame(data)
print(f"Исходные данные:\n {df}")

# -------------------------------
# 2. Подготовка данных
# -------------------------------
X = df[['сладость', 'хруст']].values
y = df['класс'].values

# Нормализация данных
Scaler = StandardScaler()
X_scaler = Scaler.fit_transform(X)

# Разделение на выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaler, y, test_size=0.3, random_state=42, stratify=y)

# -------------------------------
# 3. Реализация k-NN с нуля
# -------------------------------
class MyKNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, x, y):
        self.X_train = x
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in X:
            # Вычисляем евклидовы расстояния до всех обучающих объектов
            distances = np.linalg.norm(self.X_train - x, axis=1)
            # Получаем индексы k ближайших соседей
            k_index = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_index]
            # Выбор
            unique, counts = np.unique(k_labels, return_counts=True)
            predictions.append(unique[np.argmax(counts)])

        return np.array(predictions)


# -------------------------------
# 4. Обучение и оценка собственного классификатора
# -------------------------------
my_knn = MyKNN(3)
my_knn.fit(X_train, y_train)
y_pred_my = my_knn.predict(X_test)
acc_my = accuracy_score(y_test, y_pred_my)
print(f"\nТочность собственного k-NN (k=3): {acc_my:.2f}")
print("Отчёт классификации:\n", classification_report(y_test, y_pred_my))

# -------------------------------
# 5. Обучение и оценка sklearn k-NN
# -------------------------------
sk_knn = KNeighborsClassifier(n_neighbors=3)
sk_knn.fit(X_train, y_train)
y_pred_sk = sk_knn.predict(X_test)
acc_sk = accuracy_score(y_test, y_pred_sk)
print(f"\nТочность sklearn k-NN (k=3): {acc_sk:.2f}")
print("Отчёт классификации:\n", classification_report(y_test, y_pred_sk))

# -------------------------------
# 6. Добавление нового класса
# -------------------------------
# Добавим класс "Ягоды" с новыми примерами.
new_data = {
    'продукт': ['клубника', 'черника', 'малина'],
    'сладость': [5, 4, 5],
    'хруст': [5, 4, 4],
    'класс': ['Ягода', 'Ягода', 'Ягода']
}
df_new = pd.DataFrame(new_data)
df_extended = pd.concat([df, df_new], ignore_index=True)
print("\nРасширенный набор данных:")
print(df_extended)

X2 = df_extended[['сладость', 'хруст']].values
y2 = df_extended['класс'].values
X2_scaled = Scaler.fit_transform(X2)  # пересчитаем масштабирование на новых данных

# Повторим эксперимент
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2_scaled, y2, test_size=0.3, random_state=42, stratify=y2)

my_knn2 = MyKNN(k=3)
my_knn2.fit(X_train2, y_train2)
y_pred_my2 = my_knn2.predict(X_test2)
acc_my2 = accuracy_score(y_test2, y_pred_my2)
print(f"\nТочность собственного k-NN (k=3) на расширенных данных: {acc_my2:.2f}")
print("Отчёт классификации:\n", classification_report(y_test2, y_pred_my2))

sk_knn2 = KNeighborsClassifier(n_neighbors=3)
sk_knn2.fit(X_train2, y_train2)
y_pred_sk2 = sk_knn2.predict(X_test2)
acc_sk2 = accuracy_score(y_test2, y_pred_sk2)
print(f"\nТочность sklearn k-NN (k=3) на расширенных данных: {acc_sk2:.2f}")
print("Отчёт классификации:\n", classification_report(y_test2, y_pred_sk2))
