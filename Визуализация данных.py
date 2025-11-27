#!/usr/bin/env python
# coding: utf-8

# In[2]:


import seaborn as sns
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
print ("Линейный график (Line Chart)")
years = [2015, 2016, 2017, 2018, 2019]
sales = [100, 150, 200, 180, 250]

plt.plot(years, sales, marker='o')
plt.title("Рост продаж за 5 лет")
plt.xlabel("Год")
plt.ylabel("Продажи (млн $)")
plt.grid(True)
plt.show()
print ("Гистограмма (Histogram)")
ages = np.random.normal(35, 10, 1000)  # Средний возраст 35, отклонение 10

plt.hist(ages, bins=20, edgecolor='black')
plt.title("Распределение возраста клиентов")
plt.xlabel("Возраст")
plt.ylabel("Количество")
plt.show()
print ("Тепловая карта (Heatmap)")
data = np.random.rand(10, 10)  # Матрица 10x10 случайных чисел
sns.heatmap(data, annot=True, cmap="YlOrRd")
plt.title("Тепловая карта активности")
plt.show()
print ("Круговая диаграмма (Pie Chart)")
categories = ['Еда', 'Транспорт', 'Жилье', 'Развлечения']
expenses = [40, 20, 30, 10]
plt.pie(expenses, labels=categories, autopct='%1.1f%%')
plt.title("Распределение расходов")
plt.show()
print ("Столбчатая диаграмма (Bar Chart)")
regions = ['Север', 'Юг', 'Восток', 'Запад']
sales = [120, 90, 150, 80]
plt.bar(regions, sales, color='skyblue')
plt.title("Продажи по регионам")
plt.xlabel("Регион")
plt.ylabel("Продажи (млн $)")
plt.show()
print ("Древовидная диаграмма")
data = {
    "Категория": ["Электроника", "Электроника", "Одежда", "Одежда", "Косметика"],
    "Подкатегория": ["Смартфоны", "Ноутбуки", "Мужская", "Женская", "Парфюмерия"],
    "Продажи": [400, 250, 180, 220, 150]
}

fig = px.treemap(
    data,
    path=["Категория", "Подкатегория"],
    values="Продажи",
    color="Подкатегория",
    title="Продажи по категориям"
)
fig.show()
print ("Точечная диаграмма (Scatter Plot)")
height = np.random.normal(170, 10, 100)
weight = height * 0.4 + np.random.normal(60, 5, 100)

plt.scatter(height, weight, alpha=0.6)
plt.title("Зависимость веса от роста")
plt.xlabel("Рост (см)")
plt.ylabel("Вес (кг)")
plt.show()


# In[ ]:




