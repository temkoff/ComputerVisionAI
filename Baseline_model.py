# Baseline моделирование для проекта взвешивания свиней 

# 1. Импорты
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 2. Загрузка данных 
df = pd.read_csv('generated_dataset.csv')

# 3. Признаки и target
features = [
    'body_length',
    'withers_height',
    'mask_area',
    'camera_height',
    'curvature_angle'
]

target = 'weight'

X = df[features]
y = df[target]

# 4. Разбиение 
if 'pig_id' in df.columns:
    unique_ids = df['pig_id'].unique()

    train_ids, temp_ids = train_test_split(unique_ids, test_size=0.4, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    train_df = df[df['pig_id'].isin(train_ids)]
    val_df = df[df['pig_id'].isin(val_ids)]
    test_df = df[df['pig_id'].isin(test_ids)]

    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]
    X_test, y_test = test_df[features], test_df[target]
else:
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# 5. Pipeline
preprocessing = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# 6. Модели
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42)
}

results = []

# 7. Обучение и оценка
for name, model in models.items():
    pipeline = Pipeline([
        ('prep', preprocessing),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append([name, mae, rmse, r2])

# 8. Результаты
results_df = pd.DataFrame(results, columns=['Model', 'MAE', 'RMSE', 'R2'])
print(results_df)

# 9. Важность признаков 
rf_pipeline = Pipeline([
    ('prep', preprocessing),
    ('model', RandomForestRegressor(random_state=42))
])

rf_pipeline.fit(X_train, y_train)
importances = rf_pipeline.named_steps['model'].feature_importances_

feature_importance_df = pd.DataFrame({
    'feature': features,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print(feature_importance_df)

# 10. Дополнительно: сохраняем результаты в файл 
results_df.to_csv('results.csv', index=False)
feature_importance_df.to_csv('feature_importance.csv', index=False)
