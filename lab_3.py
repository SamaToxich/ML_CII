import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import time

# Генерация синтетических данных (одномерный случай)
np.random.seed(42)
X_simple = np.random.rand(100, 1) * 10  # признак от 0 до 10
y_simple = 2.5 * X_simple.squeeze() + 1.2 + np.random.randn(100) * 1.5  # y = 2.5x + 1.2 + шум

# Для многомерного примера используем встроенный набор данных (диабет)
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
X_multi = diabetes.data
y_multi = diabetes.target

# Разделение на обучающую и тестовую выборки
X_simple_train, X_simple_test, y_simple_train, y_simple_test = train_test_split(
    X_simple, y_simple, test_size=0.2, random_state=42
)
X_multi_train, X_multi_test, y_multi_train, y_multi_test = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42
)


class LinearRegression:
    def __init__(self, fit_intercept):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        if self.fit_intercept:
            X_design = np.hstack([np.ones((X.shape[0], 1)), X])
        else:
            X_design = X

        XtX = X_design.T @ X_design
        Xty = X_design.T @ y
        self.weights = np.linalg.solve(XtX, Xty)

        if self.fit_intercept:
            self.intercept_ = self.weights[0]
