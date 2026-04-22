import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../data/cardiovascular_data.csv')

print("\nПервые 5 строк:")
print(df.head())

X, y = df.drop('disease', axis=1), df['disease']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression(class_weight='balanced')
model.fit(x_train, y_train)

y_pred = (model.predict(x_test).round()).astype(int)

print(f"\nТочность: {accuracy_score(y_test, y_pred):.3f}")
print(f'\nМатрица ошибок:\n{confusion_matrix(y_test, y_pred)}')
print(f'\nОтчёт по классификации:\n{classification_report(y_test, y_pred)}')
