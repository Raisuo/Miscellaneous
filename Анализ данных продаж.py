#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
# 1. Подготовка данных
# Данные о продажах по месяцам и категориям
data = {
'Месяц': ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь', 'Июль', 'Август', 'Сентябрь',
'Октябрь', 'Ноябрь', 'Декабрь'],
'Общие_продажи': [120, 150, 180, 200, 220, 210, 230, 250, 240, 260, 280, 300],
'Электроника': [50, 60, 70, 80, 90, 85, 95, 100, 95, 110, 120, 130],
'Одежда': [40, 50, 60, 70, 80, 75, 85, 90, 85, 95, 100, 110],
'Книги': [30, 40, 50, 50, 50, 50, 50, 60, 60, 55, 60, 60]
}
df_monthly = pd.DataFrame(data)
# Данные о продажах по регионам
regions_data = {
'Регион': ['Москва', 'Санкт-Петербург', 'Новосибирск', 'Екатеринбург'],
'Продажи': [900, 700, 500, 400]
}
df_regions = pd.DataFrame(regions_data)
# Проверка данных на пропуски
print("Проверка данных на пропуски (по месяцам):")
print(df_monthly.isnull().sum())
print("\nПроверка данных на пропуски (по регионам):")
print(df_regions.isnull().sum())

# 2. Визуализация трендов продаж по месяцам (линейный график)
plt.figure(figsize=(10, 6))
plt.plot(df_monthly['Месяц'], df_monthly['Общие_продажи'], marker='o', color='blue', linewidth=2)
plt.title('Динамика продаж по месяцам', fontsize=14, fontweight='bold')
plt.xlabel('Месяц', fontsize=12)
plt.ylabel('Продажи (тыс. руб.)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('monthly_sales_trend.png')
plt.show()

# 3. Анализ продаж по категориям товаров (столбчатая диаграмма с накоплением)
plt.figure(figsize=(10, 6))
plt.bar(df_monthly['Месяц'], df_monthly['Электроника'], label='Электроника', color='skyblue')
plt.bar(df_monthly['Месяц'], df_monthly['Одежда'], bottom=df_monthly['Электроника'],
label='Одежда', color='lightgreen')
plt.bar(df_monthly['Месяц'], df_monthly['Книги'], bottom=df_monthly['Электроника'] +
df_monthly['Одежда'], label='Книги', color='lightcoral')
plt.title('Продажи по категориям товаров по месяцам', fontsize=14, fontweight='bold')
plt.xlabel('Месяц', fontsize=12)
plt.ylabel('Продажи (тыс. руб.)', fontsize=12)
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('category_sales_bar.png')
plt.show()

# 4. Сравнение продаж по регионам (круговая диаграмма)
plt.figure(figsize=(8, 8))
plt.pie(df_regions['Продажи'], labels=df_regions['Регион'], autopct='%1.1f%%', colors=['gold',
'lightblue', 'lightgreen', 'lightcoral'])
plt.title('Доля продаж по регионам за год', fontsize=14, fontweight='bold')
plt.savefig('region_sales_pie.png')
plt.show()

# 5. Создание простого дашборда (объединение всех графиков)
fig = plt.figure(figsize=(18, 12))
fig.suptitle('Дашборд продаж интернет-магазина', fontsize=16, fontweight='bold')
# Линейный график (тренд продаж по месяцам)
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(df_monthly['Месяц'], df_monthly['Общие_продажи'], marker='o', color='blue', linewidth=2)
ax1.set_title('Динамика продаж по месяцам')
ax1.set_xlabel('Месяц')
ax1.set_ylabel('Продажи (тыс. руб.)')

# Правильный способ установить метки:
ax1.set_xticks(range(len(df_monthly['Месяц'])))  # Устанавливаем тики для каждого месяца
ax1.set_xticklabels(df_monthly['Месяц'], rotation=45, ha='right') #ha для выравнивания

ax1.grid(True, linestyle='--', alpha=0.7)

# Столбчатая диаграмма (категории товаров)
ax2 = fig.add_subplot(2, 2, 2)
ax2.bar(df_monthly['Месяц'], df_monthly['Электроника'], label='Электроника', color='skyblue')
ax2.bar(df_monthly['Месяц'], df_monthly['Одежда'], bottom=df_monthly['Электроника'],
label='Одежда', color='lightgreen')
ax2.bar(df_monthly['Месяц'], df_monthly['Книги'], bottom=df_monthly['Электроника'] +
df_monthly['Одежда'], label='Книги', color='lightcoral')
ax2.set_title('Продажи по категориям товаров')
ax2.set_xlabel('Месяц')
ax2.set_ylabel('Продажи (тыс. руб.)')

# Правильный способ установить метки:
ax2.set_xticks(range(len(df_monthly['Месяц']))) # Устанавливаем тики для каждого месяца
ax2.set_xticklabels(df_monthly['Месяц'], rotation=45, ha='right') #ha для выравнивания

ax2.legend()
ax2.grid(True, axis='y', linestyle='--', alpha=0.7)

# Круговая диаграмма (регионы)
ax3 = fig.add_subplot(2, 2, 3)
ax3.pie(df_regions['Продажи'], labels=df_regions['Регион'], autopct='%1.1f%%', colors=['gold',
'lightblue', 'lightgreen', 'lightcoral'])
ax3.set_title('Доля продаж по регионам')

plt.tight_layout()
plt.savefig('sales_dashboard.png')
plt.show()
# 1. Функция для вычисления роста продаж в процентах
def calculate_sales_growth(df, sales_column):

    sales_growth = df[sales_column].pct_change() * 100
    return sales_growth

# 2. Функция для определения лидеров продаж по месяцам
def find_sales_leaders(df, category_columns):
    leaders = {}
    for category in category_columns:
        leaders[category] = df['Месяц'][df[category].idxmax()]

    leaders_df = pd.DataFrame.from_dict(leaders, orient='index', columns=['Лидер'])
    leaders_df.index.name = 'Категория'
    return leaders_df


# 3. Функция для определения наличия сезонности (с использованием декомпозиции временных рядов)
def check_seasonality_decomposition(df, sales_column):
    # Преобразуем столбец 'Месяц' в индекс, чтобы использовать данные как временной ряд
    df = df.set_index('Месяц')

    # Декомпозиция временного ряда
    try:
        decomposition = seasonal_decompose(df[sales_column], model='additive', period=6) # period=6 предполагает полугодовую сезонность
        #decomposition = seasonal_decompose(df[sales_column], model='additive', period=12) # period=12 предполагает годовую сезонность

    except Exception as e:
        return f"Ошибка при декомпозиции временного ряда: {e}. Возможно, недостаточно данных."

    # Визуализация компонентов (опционально, но рекомендуется)
    plt.figure(figsize=(12, 8))
    plt.subplot(411)
    plt.plot(df[sales_column], label='Исходные данные')
    plt.legend(loc='upper left')
    plt.subplot(412)
    plt.plot(decomposition.trend, label='Тренд')
    plt.legend(loc='upper left')
    plt.subplot(413)
    plt.plot(decomposition.seasonal, label='Сезонность')
    plt.legend(loc='upper left')
    plt.subplot(414)
    plt.plot(decomposition.resid, label='Остаток')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


    # Анализ сезонной компоненты
    seasonal_strength = decomposition.seasonal.abs().mean() / df[sales_column].mean()
    # Другой вариант оценки - сравнение дисперсии сезонной компоненты с дисперсией исходного ряда

    if seasonal_strength > 0.1:  # 0.1 - произвольный порог.  Нужно выбирать исходя из контекста.
        return "Выявлена сезонность на основе декомпозиции временного ряда."
    else:
        return "Сезонность не выражена на основе декомпозиции временного ряда."
# 1. Рост общих продаж
growth = calculate_sales_growth(df_monthly, 'Общие_продажи')
print("\nРост общих продаж по месяцам (в процентах):")
print(growth)

# 2. Лидеры продаж по категориям
category_columns = ['Электроника', 'Одежда', 'Книги']
leaders_df = find_sales_leaders(df_monthly, category_columns)
print("\nЛидеры продаж по категориям:")
print(leaders_df)

# 3. Проверка на сезонность
seasonality_check = check_seasonality_decomposition(df_monthly, 'Общие_продажи')
print("\nНаличие сезонности (с использованием декомпозиции временных рядов):")
print(seasonality_check)
print ("""Продажи демонстрируют устойчивый рост с января по декабрь, однако сезонность не выявлена. Наибольший доход приносит электроника с октября
по декабрь. В это время повышенные продажи одежды. Книги имеют стабильно низкие продажи в течении года. Москва является лидером по продажам, 
составляя 36% от общего объема, что указывает на необходимость усиления маркетинга в других регионах. Продажи изделий демонстрируют рост, прерываемый
только в мае и сентябре. После чего на январь следующего года приходится резкий спад.""")


# In[ ]:




