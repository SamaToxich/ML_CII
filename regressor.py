"""
Задача: Регрессия (предсказание вероятности сердечно-сосудистого заболевания)
Датасет: cardiovascular_data.csv (1500 записей, 7 признаков, бинарная целевая переменная)
Модель: XGBRegressor с objective='reg:logistic'
Цель: Получить вероятности заболевания и оценить качество предсказаний
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

df = pd.read_csv('cardiovascular_data.csv')
print("Размер датасета:", df.shape)
print("\nПервые 5 строк:")
print(df.head())
print("\nБаланс классов:")
print(df['disease'].value_counts(normalize=True))

X = df.drop('disease', axis=1)
y = df['disease']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nРазмер обучающей выборки: {x_train.shape}")
print(f"Размер тестовой выборки: {x_test.shape}")

# Базовая модель
model = XGBRegressor(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.01,
    objective='reg:logistic',
    random_state=42,
    eval_metric='logloss'
)

# Сетка параметров
#param_grid = {
#    'n_estimators': [100, 200, 300],
#    'max_depth': [3, 5, 7],
#    'learning_rate': [0.01, 0.05, 0.1],
#    'subsample': [0.7, 0.8, 1.0],
#    'colsample_bytree': [0.7, 0.8, 1.0]}

# GridSearchCV с регрессионной метрикой (отрицательная MSE)
#grid = GridSearchCV(
#    estimator=model,
#    param_grid=param_grid,
#    scoring='neg_mean_squared_error',  # максимизируем -MSE
#    cv=5,                              # 5-кратная кросс-валидация
#    n_jobs=-1,
#    verbose=1)

#print("\nНачало подбора гиперпараметров...")
#grid.fit(X_train, y_train)

# Предсказание вероятностей (регрессионный выход)
model.fit(x_train, y_train)
y_proba = model.predict(x_test)

# Метрики регрессии
mae = mean_absolute_error(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

print("""
--- Интерпретация ---
MAE = {:.3f} — средняя абсолютная ошибка предсказания вероятности составляет {:.1f}%.
ROC-AUC = {:.3f} — модель различает классы лучше случайного.
""".format(mae, mae*100, roc_auc))