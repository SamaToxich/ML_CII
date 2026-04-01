import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb

# 1. Создаём синтетический датасет
np.random.seed(42)
n_samples = 10000
feature1 = np.random.normal(5, 3, n_samples)
feature2 = np.random.normal(100, 50, n_samples)
feature3 = np.random.normal(10, 3, n_samples)

target = ((feature1 > 5.5) & (feature2 > 110)).astype(int)
noise = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
target = np.logical_xor(target, noise).astype(int)

df = pd.DataFrame({
    'Лейкоцыты': feature1,
    'Глюкоза': feature2,
    'Crp': feature3,
    'Цель': target
})

print(df)

# 2. Подготовка
X = df.drop('Цель', axis=1)
y = df['Цель']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Обучение XGBoost
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# 4. Оценка
y_pred = model.predict(X_test)
print(f"\nТочность: {accuracy_score(y_test, y_pred):.4f}")
print("\nМатрица ошибок:")
print(*confusion_matrix(y_test, y_pred))
print("\nОтчёт по классификации:")
print(classification_report(y_test, y_pred, target_names=['Здоров', 'Болен']))