import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/cardiovascular_data.csv')

X, y = df.drop('disease', axis=1), df['disease']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


class MyLogisticRegressionMulti:
    def __init__(self, lr=0.1, n_iteration=1000, random_state=None):
        self.lr=lr
        self.n_iter=n_iteration
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.classes_ = None

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        if self.random_state:
            np.random.seed(self.random_state)

        self.weights = np.random.randn(n_features, n_classes) * 0.01
        self.bias = np.zeros(n_classes)

        # Преобразуем y в one-hot
        y_onehot = np.zeros((n_samples, n_classes))

        for i, cls in enumerate(self.classes_):
            y_onehot[:,i] = (y == cls).astype(int)

        for _ in range(self.n_iter):
            linear = X @ self.weights + self.bias

            y_pred = self.softmax(linear)

            dw = X.T @ (y_pred - y_onehot)
            db = np.sum(y_pred-y_onehot, axis=0)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self,X):
        linear = X @ self.weights + self.bias
        return self.softmax(linear)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# Многоклассовая на тех же данных
my_multi = MyLogisticRegressionMulti()
my_multi.fit(X_train, y_train)
y_pred_multi_my = my_multi.predict(X_test)
acc_multi_my = accuracy_score(y_test, y_pred_multi_my)

sk_multi = LogisticRegression(class_weight='balanced')
sk_multi.fit(X_train, y_train)
y_pred_multi_sk = sk_multi.predict(X_test)
acc_multi_sk = accuracy_score(y_test, y_pred_multi_sk)

print(f"MyLogReg (softmax) точность: {acc_multi_my:.2f}")
print(classification_report(y_test, y_pred_multi_my))
print(f"Sklearn (multinomial) точность: {acc_multi_sk:.2f}")
print(classification_report(y_test, y_pred_multi_sk))
