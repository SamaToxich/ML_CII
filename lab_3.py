import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Загрузка данных
iris = load_iris()
X, y = iris.data, iris.target

# Масштабирование
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

class MyLogisticRegression:
    def __init__(self, learning_rate=0.1, n_iters=1000, с=1.0):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.с = 1/с
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        # Градиентный спуск
        for _ in range(self.n_iters):
            # Линейная комбинация
            linear_model = X @ self.weights + self.bias

            y_pred = self._sigmoid(linear_model)

            # Градиенты
            dw = (X.T @ (y_pred - y)) + 2 * self.с * self.weights # L2
            db = np.sum(y_pred - y)

            # Обновление параметров
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        linear_model = X @ self.weights + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)


class OneVsRestLogistic:
    def __init__(self, learning_rate=0.1, n_iters=1000, с=1):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.с = с
        self.models = []        # список бинарных классификаторов
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        # Для каждого класса создаём и обучаем бинарную модель
        for c in self.classes_:
            # Преобразуем метки: 1 для класса c, 0 для всех остальных
            y_binary = (y == c).astype(int)
            model = MyLogisticRegression(learning_rate=self.lr, n_iters=self.n_iters, с=self.с)
            model.fit(X, y_binary)
            self.models.append(model)

    def predict_proba(self, X):
        # Возвращает вероятности для каждого класса (матрица n_samples x n_classes)
        probas = np.column_stack([model.predict_proba(X) for model in self.models])
        return probas

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

my_lr = OneVsRestLogistic(learning_rate=0.1, n_iters=100, с=1000)
my_lr.fit(X_train, y_train)
y_pred_my = my_lr.predict(X_test)
print(f"Моя softmax точность: {accuracy_score(y_test, y_pred_my):.3f}")

sk_lr = LogisticRegression()
sk_lr.fit(X_train, y_train)
y_pred_sk = sk_lr.predict(X_test)
print(f"Sklearn softmax точность: {accuracy_score(y_test, y_pred_sk):.3f}")