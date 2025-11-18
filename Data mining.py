#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import requests
from bs4 import BeautifulSoup
from io import StringIO

# ======================
# 1. Сбор данных (Data Collection)
# ======================

print("=== Этап 1: Сбор данных ===")

# Вариант 1: Загрузка данных из CSV (имитация базы данных)
print("\n1.1 Загрузка данных из CSV файла...")
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
data = pd.read_csv(url)
print(f"Данные успешно загружены. Размер датасета: {data.shape}")

# Вариант 2: Получение данных через API (пример с открытым API)
print("\n1.2 Получение данных через API...")
try:
    api_url = "https://api.github.com/repos/pandas-dev/pandas"
    response = requests.get(api_url)
    api_data = response.json()
    print(f"Данные из GitHub API: Репозиторий '{api_data['name']}' имеет {api_data['stargazers_count']} звезд")
except:
    print("Не удалось получить данные через API")

# Вариант 3: Веб-скрейпинг (пример с википедии)
print("\n1.3 Веб-скрейпинг данных...")
try:
    wiki_url = "https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)"
    response = requests.get(wiki_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    tables = soup.find_all('table', {'class': 'wikitable'})
    gdp_table = pd.read_html(StringIO(str(tables[0])))[0]
    print("Данные GDP с Википедии (первые 5 строк):")
    print(gdp_table.head())
except:
    print("Не удалось выполнить веб-скрейпинг")

# ======================
# 2. Очистка данных (Data Cleaning)
# ======================

print("\n=== Этап 2: Очистка данных ===")

# Создадим копию данных для очистки
clean_data = data.copy()

# 2.1 Проверка на пропущенные значения
print("\n2.1 Проверка на пропущенные значения:")
print(clean_data.isnull().sum())

# 2.2 Заполнение пропущенных значений
print("\n2.2 Заполнение пропущенных значений...")
clean_data['age'].fillna(clean_data['age'].median(), inplace=True)
clean_data['embarked'].fillna(clean_data['embarked'].mode()[0], inplace=True)
clean_data.drop('deck', axis=1, inplace=True)  # Удаление столбца с большим количеством пропусков

# 2.3 Удаление дубликатов
print("\n2.3 Удаление дубликатов...")
clean_data.drop_duplicates(inplace=True)

# 2.4 Исправление форматов данных
print("\n2.4 Исправление форматов данных...")
clean_data['sex'] = clean_data['sex'].astype('category')
clean_data['embarked'] = clean_data['embarked'].astype('category')

print("\nРезультат очистки:")
print(clean_data.info())

# ======================
# 3. Исследовательский анализ данных (EDA)
# ======================

print("\n=== Этап 3: Исследовательский анализ данных ===")

# 3.1 Описательная статистика
print("\n3.1 Описательная статистика:")
print(clean_data.describe())

# 3.2 Визуализация распределений
print("\n3.2 Визуализация данных...")

plt.figure(figsize=(15, 10))

# Распределение возрастов
plt.subplot(2, 2, 1)
sns.histplot(clean_data['age'], bins=30, kde=True)
plt.title('Распределение возраста пассажиров')

# Ящик с усами для стоимости билетов
plt.subplot(2, 2, 2)
sns.boxplot(x='pclass', y='fare', data=clean_data)
plt.title('Распределение стоимости билетов по классам')

# Количество выживших по полу
plt.subplot(2, 2, 3)
sns.countplot(x='survived', hue='sex', data=clean_data)
plt.title('Количество выживших по полу')

# Тепловая карта корреляций
plt.subplot(2, 2, 4)
numeric_data = clean_data.select_dtypes(include=['number'])
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
plt.title('Тепловая карта корреляций')

plt.tight_layout()
plt.show()

# ======================
# 4. Преобразование данных (Data Transformation)
# ======================

print("\n=== Этап 4: Преобразование данных ===")

# 4.1 Создание новых признаков
print("\n4.1 Создание новых признаков...")
clean_data['family_size'] = clean_data['sibsp'] + clean_data['parch'] + 1
clean_data['is_alone'] = (clean_data['family_size'] == 1).astype(int)

# 4.2 Кодирование категориальных переменных
print("\n4.2 Кодирование категориальных переменных...")
clean_data = pd.get_dummies(clean_data, columns=['sex', 'embarked'], drop_first=True)

# 4.3 Нормализация числовых признаков
print("\n4.3 Нормализация числовых признаков...")
scaler = StandardScaler()
numeric_cols = ['age', 'fare', 'family_size']
clean_data[numeric_cols] = scaler.fit_transform(clean_data[numeric_cols])

print("\nДанные после преобразования:")
print(clean_data.head())

# ======================
# 5. Анализ и моделирование (Data Analysis & Modeling)
# ======================

print("\n=== Этап 5: Анализ и моделирование ===")

# 5.1 Подготовка данных для моделирования
print("\n5.1 Подготовка данных для моделирования...")
X = clean_data.drop(['survived', 'alive', 'class', 'who', 'adult_male', 'embark_town'], axis=1, errors='ignore')
y = clean_data['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5.2 Обучение модели
print("\n5.2 Обучение модели...")
model = LinearRegression()
model.fit(X_train, y_train)

# 5.3 Оценка модели
print("\n5.3 Оценка модели...")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Среднеквадратичная ошибка (MSE): {mse:.4f}")

# ======================
# 6. Визуализация и отчетность (Data Visualization & Reporting)
# ======================

print("\n=== Этап 6: Визуализация и отчетность ===")

# 6.1 Визуализация важности признаков
print("\n6.1 Визуализация важности признаков...")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.coef_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Важность признаков в модели')
plt.show()

# 6.2 Создание отчета
print("\n6.2 Создание отчета...")
report = f"""
Отчет по анализу данных:
1. Исходные данные: {data.shape[0]} строк, {data.shape[1]} столбцов
2. После очистки: {clean_data.shape[0]} строк, {clean_data.shape[1]} столбцов
3. Качество модели (MSE): {mse:.4f}
4. Самые важные признаки:
{feature_importance.head(3).to_string(index=False)}
"""

print(report)

# Сохранение отчета в файл
with open('data_analysis_report.txt', 'w') as f:
    f.write(report)
print("Отчет сохранен в файл 'data_analysis_report.txt'")

print("\nАнализ данных завершен!")


# In[ ]:




