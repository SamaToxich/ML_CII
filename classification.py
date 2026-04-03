import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('medical_data.csv')

print("\nПервые 5 строк:")
print(df.head())

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

model = KNeighborsClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"\nТочность: {accuracy_score(y_test, y_pred):.4f}")
print(f'\nМатрица ошибок:\n{confusion_matrix(y_test, y_pred)}')
print(f'\nОтчёт по классификации:\n{classification_report(y_test, y_pred, target_names=['Здоров', 'Болен'])}')