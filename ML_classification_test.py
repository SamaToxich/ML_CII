import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

df = pd.read_csv('medical_data.csv')

x = df.drop('diagnosis', axis=1)
y = df['diagnosis']

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

print(df.describe())

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42, stratify=y)

model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(f"\nТочность: {accuracy_score(y_test, y_pred):.4f}\n")
print(f"Матрица ошибок:\n {confusion_matrix(y_test, y_pred)}")
print(f'\nОтчёт по классификациям:\n {classification_report(y_test, y_pred, target_names=['Здоров', 'Болен'])}')
