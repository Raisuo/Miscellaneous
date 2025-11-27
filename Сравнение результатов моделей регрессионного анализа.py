#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
ПРОДВИНУТЫЙ РЕГРЕССИОННЫЙ АНАЛИЗ В PYTHON
Сравнение различных алгоритмов регрессии с визуализацией результатов
"""

# Импорт необходимых библиотек
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# =============================================
# 1. ГЕНЕРАЦИЯ И ПОДГОТОВКА ДАННЫХ
# =============================================

# Генерация синтетических данных
np.random.seed(100)
n_samples = 400
n_features = 3

# Создание матрицы признаков
X = np.random.rand(n_samples, n_features) * 10

# Создание целевой переменной с нелинейными зависимостями
y = (2.5 * X[:, 0] +
     1.7 * np.sin(X[:, 1]) -
     3.2 * (X[:, 2] > 5).astype(int) +
     np.random.normal(0, 2, n_samples))

# Преобразование в DataFrame для наглядности
features = [f'Feature_{i}' for i in range(n_features)]
df = pd.DataFrame(X, columns=features)
df['Target'] = y

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Масштабирование данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================
# 2. РЕГРЕССИОННЫЕ МОДЕЛИ
# =============================================

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

# Инициализация моделей
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Support Vector Regression': SVR(kernel='rbf', C=1.0, epsilon=0.1),
    'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(50, 20), activation='relu',
                                   solver='adam', max_iter=500, random_state=42)
}
"""
Формирует дубликат (скрытый слой?), пользуясь объявленным количеством нейронов, активирует ключом 'relu' по умолчанию,
время обсчёта с весовым критерием 5 и 2?,
adam отвечает за сортировку.
Чем больше слоёв тем выше точность
"""
# Обучение и предсказание для каждой модели
results = {}
for name, model in models.items():
    # Для некоторых моделей используем масштабированные данные, для других - нет
    if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression',
                'Support Vector Regression', 'Neural Network']:
        X_tr = X_train_scaled
        X_te = X_test_scaled
    else:
        X_tr = X_train
        X_te = X_test

    # Обучение модели
    model.fit(X_tr, y_train)

    # Предсказание на тестовых данных
    y_pred = model.predict(X_te)

    # Сохранение результатов
    results[name] = {
        'model': model,
        'predictions': y_pred,
        'mse': mean_squared_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }

# =============================================
# 3. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# =============================================

# Создаем фигуру для графиков
plt.figure(figsize=(15, 10))

# График 1: Сравнение метрик
plt.subplot(2, 2, 1)
mse_values = [results[name]['mse'] for name in results]
r2_values = [results[name]['r2'] for name in results]
x = np.arange(len(models))
width = 0.35

plt.bar(x - width / 2, mse_values, width, label='MSE')
plt.bar(x + width / 2, r2_values, width, label='R2 Score')
plt.xticks(x, list(results.keys()), rotation=45, ha='right')
plt.title('Сравнение метрик моделей')
plt.ylabel('Значение метрики')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# График 2: Фактические vs Предсказанные значения (лучшая модель)
best_model_name = min(results, key=lambda x: results[x]['mse'])
y_pred_best = results[best_model_name]['predictions']
plt.subplot(2, 2, 2)
plt.scatter(y_test, y_pred_best, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r')
plt.xlabel('Фактические значения')
plt.ylabel('Предсказанные значения')
plt.title(f'Фактические vs Предсказанные ({best_model_name})')
plt.grid(True, linestyle='--', alpha=0.7)

# График 3: Важность признаков (для Random Forest)
plt.subplot(2, 2, 3)
if hasattr(results['Random Forest']['model'], 'feature_importances_'):
    importances = results['Random Forest']['model'].feature_importances_
    plt.barh(features, importances)
    plt.title('Важность признаков (Random Forest)')
    plt.xlabel('Важность')
    plt.grid(True, linestyle='--', alpha=0.7)

# График 4: Остатки для лучшей модели
plt.subplot(2, 2, 4)
residuals = y_test - y_pred_best
plt.scatter(y_pred_best, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Предсказанные значения')
plt.ylabel('Остатки')
plt.title('График остатков (лучшая модель)')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# =============================================
# 4. ВЫВОД РЕЗУЛЬТАТОВ
# =============================================

# Вывод таблицы с результатами
print("\n{:<25} {:<15} {:<15}".format('Модель', 'MSE', 'R2 Score'))
print("-" * 50)
for name in results:
    print("{:<25} {:<15.3f} {:<15.3f}".format(
        name, results[name]['mse'], results[name]['r2']))

# Вывод коэффициентов для линейных моделей
print("\nКоэффициенты линейных моделей:")
linear_models = ['Linear Regression', 'Ridge Regression', 'Lasso Regression']
for name in linear_models:
    if hasattr(results[name]['model'], 'coef_'):
        print(f"\n{name}:")
        for i, coef in enumerate(results[name]['model'].coef_):
            print(f"{features[i]}: {coef:.3f}")
        if hasattr(results[name]['model'], 'intercept_'):
            print(f"Intercept: {results[name]['model'].intercept_:.3f}")

# Вывод информации о лучшей модели
print(
    f"\nЛучшая модель: {best_model_name} (MSE: {results[best_model_name]['mse']:.3f}, R2: {results[best_model_name]['r2']:.3f})")


# In[ ]:




