import pandas
from sklearn.metrics import classification_report, r2_score, roc_auc_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV

df = pandas.read_csv('cardiovascular_data.csv')

print(df.head())

x = df.drop('disease', axis=1)
y = df['disease']

print()
print(y.value_counts(normalize=True))

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42, stratify=y)

model = XGBRegressor(random_state=42, objective='reg:logistic')

param_grid = {
    'max_depth': [1, 3, 5],
    'learning_rate': [0.001, 0.01, 0.1],
    'n_estimators': [100, 200, 300],
}

grid = GridSearchCV(
    model,
    param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1,
    verbose=1
)
grid.fit(x_train, y_train)

print("\nBest params:", grid.best_params_)

best_model = grid.best_estimator_
y_proba = best_model.predict(x_test)

print(f"MAE: {mean_absolute_error(y_test, y_proba):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")