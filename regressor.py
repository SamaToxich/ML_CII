import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

df = pd.read_csv('cardiovascular_data.csv')
print("Размер датасета:", df.shape)
print("\nПервые 5 строк:")
print(df.head())
print("\nБаланс классов:")
print(df['disease'].value_counts(normalize=True))

iris = load_iris()
#X, y = iris.data, iris.target
X, y = df.drop('disease', axis=1), df['disease']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nРазмер обучающей выборки: {x_train.shape}")
print(f"Размер тестовой выборки: {x_test.shape}\n")

# Базовая модель
#model = XGBRegressor()
model = LogisticRegression()

model.fit(x_train, y_train)
y_pred = (model.predict(x_test).round()).astype(int)

print(f'Тестовая выборка: {y_test.values}')
print(f'Предсказания:     {y_pred}')

print(f"\nТочность: {accuracy_score(y_test, y_pred):.3f}")