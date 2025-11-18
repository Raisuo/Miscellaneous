#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Импортируем необходимые библиотеки
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm  #  Для теста Голдфелда-Куандта
# Данные о количестве часов подготовки и итоговых баллах
hours = np.array([2, 4, 5, 4, 5, 7, 8, 6, 9, 10]).reshape(-1, 1) # Независимая переменная
scores = np.array([50, 55, 60, 58, 62, 70, 75, 65, 80, 85]) # Зависимая переменная
# 1. Расчет статистических показателей для обеих переменных
# Часы подготовки
mean_hours = np.mean(hours) # Среднее значение
std_hours = np.std(hours, ddof=1) # Стандартное отклонение (выборочное)
size_hours = len(hours) # Размер выборки
# Итоговые баллы
mean_scores = np.mean(scores)
std_scores = np.std(scores, ddof=1)
size_scores = len(scores)
# Вывод статистических показателей
print("Статистические показатели:")
print("Количество часов подготовки:")
print(f"Среднее значение: {mean_hours:.2f}")
print(f"Стандартное отклонение: {std_hours:.2f}")
print(f"Размер выборки: {size_hours}")
print("\nИтоговые баллы:")
print(f"Среднее значение: {mean_scores:.2f}")
print(f"Стандартное отклонение: {std_scores:.2f}")
print(f"Размер выборки: {size_scores}")
# 2. Линейный регрессионный анализ
# Создаем модель линейной регрессии
model = LinearRegression()
model.fit(hours, scores)
# Получаем коэффициенты регрессии
slope = model.coef_[0] # Наклон (коэффициент при X)
intercept = model.intercept_ # Пересечение с осью Y
r2 = r2_score(scores, model.predict(hours)) # Коэффициент детерминации R²
# Вывод коэффициентов регрессии
print("\nРезультаты линейной регрессии:")
print(f"Наклон (коэффициент при X): {slope:.2f}")
print(f"Пересечение с осью Y: {intercept:.2f}")
print(f"Коэффициент детерминации (R²): {r2:.2f}")
# 3. Проверка статистической значимости модели
# Используем t-тест для наклона (slope)
# Вычисляем стандартную ошибку наклона и p-значение
n = len(hours)
y_pred = model.predict(hours)
residuals = scores - y_pred
ss_tot = np.sum((scores - mean_scores) ** 2)
ss_res = np.sum(residuals ** 2)
se_slope = np.sqrt(ss_res / (n - 2)) / np.sqrt(np.sum((hours.flatten() - mean_hours) ** 2))
t_stat = slope / se_slope
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2)) # Двусторонний тест
alpha = 0.05 # Уровень значимости
# Вывод результатов проверки значимости
print("\nПроверка статистической значимости модели:")
print(f"t-статистика для наклона: {t_stat:.2f}")
print(f"p-значение: {p_value:.4f}")
if p_value < alpha:
 print("p-значение < 0.05: Модель статистически значима.")
else:
 print("p-значение >= 0.05: Модель не является статистически значимой.")
# 6.3. Проверка данных на выбросы (шум)
# Используем метод z-оценок
z_scores = np.abs(stats.zscore(scores))
threshold = 2  # Порог для выявления выбросов (обычно 2 или 3)
outliers = np.where(z_scores > threshold)[0]  # Индексы выбросов
print("\nАнализ выбросов (шума):")
if len(outliers) > 0:
    print("Обнаружены выбросы в данных:")
    for i in outliers:
        print(f"  Наблюдение {i+1}: Часы = {hours[i][0]}, Баллы = {scores[i]}, Z-оценка = {z_scores[i]:.2f}")
    #  (Опционально) Удаление выбросов из данных.  ВНИМАНИЕ:  Это изменяет ваши исходные данные!
    #  hours = np.delete(hours, outliers, axis=0)
    #  scores = np.delete(scores, outliers)
    #  print("Выбросы удалены из данных (если нужно, раскомментируйте эти строки).")
else:
    print("Выбросы не обнаружены.")

# 5. Анализ остатков (residuals)
#   (Уже вычислены в п.3, но пересчитываем, чтобы было понятнее)
y_pred = model.predict(hours)  # Пересчитываем, если данные были изменены
residuals = scores - y_pred
# 6.4 Тест Голдфелда-Куандта на гетероскедастичность
X = sm.add_constant(hours)  # Добавляем константу для statsmodels
model_sm = sm.OLS(scores, X)  #  Создаем OLS модель из statsmodels
results = model_sm.fit()
# Выполняем тест Голдфелда-Куандта.  `alternative='increasing'` предполагает, что дисперсия увеличивается с ростом hours.
#  `split=mean_hours` задает точку, вокруг которой делится выборка.
gq_test = sm.stats.het_goldfeldquandt(results.resid, results.model.exog, alternative='increasing', split=int(mean_hours))
print("\nТест Голдфелда-Куандта на гетероскедастичность:")
print(f"  F-статистика: {gq_test[0]:.4f}")
print(f"  p-значение: {gq_test[1]:.4f}")
if gq_test[1] < alpha:
    print("  p-значение < 0.05: Есть основания полагать наличие гетероскедастичности.")
else:
    print("  p-значение >= 0.05: Нет достаточных оснований полагать наличие гетероскедастичности.")
# 4. Визуализация данных
# Настройка фигуры для диаграммы рассеяния
plt.figure(figsize=(8, 6))
# 4.1 Диаграмма рассеяния (scatter plot)
plt.scatter(hours, scores, color='blue', alpha=0.6, label='Данные студентов')
# 4.2 Добавление линии регрессии
plt.plot(hours, y_pred, color='red', linestyle='--', label='Линия регрессии')
# Настройка графика
plt.title('Регрессионный анализ: Часы подготовки и итоговые баллы')
plt.xlabel('Часы подготовки')
plt.ylabel('Итоговые баллы')
plt.grid(True)
plt.legend()
#6.1 график с примером анализа остатков
plt.figure(figsize=(8, 6), num=2)  # num=2 для указания номера фигуры (для сохранения)
plt.scatter(hours, residuals, color='purple', alpha=0.6)
plt.axhline(y=0, color='black', linestyle='--')
plt.title('График остатков')
plt.xlabel('Часы подготовки')
plt.ylabel('Остатки')
plt.grid(True)

# 6.3 Анализ остатков (добавление к графику остатков)
plt.figure(2) # Выбираем 2-ю фигуру (график остатков)
sns.kdeplot(residuals, fill=True, alpha=0.3, color='green', label='Плотность остатков') # добавляем график плотности
plt.legend()
# 6.2 Сохранение графиков в файл по желанию пользователя
save_fig = input("Сохранить графики в файл? (да/нет): ").strip().lower()
if save_fig in ['да', 'д', 'yes', 'y', 'давай', 'go']:
    save_location = input("Укажите путь для сохранения файлов (или нажмите Enter для сохранения в текущей директории): ").strip()

    # Сохранение первого графика (диаграмма рассеяния)
    filename1 = 'regression_analysis.png'
    if save_location:
        full_filename1 = os.path.join(save_location, filename1)
    else:
        full_filename1 = filename1
    plt.figure(1)  #  Выбираем первую фигуру, чтобы сохранить её
    plt.savefig(full_filename1)
    print(f"График 1 (диаграмма рассеяния) сохранён в файл '{full_filename1}'")

    # Сохранение второго графика (остатки)
    filename2 = 'residuals_plot.png'
    if save_location:
        full_filename2 = os.path.join(save_location, filename2)
    else:
        full_filename2 = filename2
    plt.figure(2) #  Выбираем вторую фигуру, чтобы сохранить её
    plt.savefig(full_filename2)
    print(f"График 2 (остатки) сохранён в файл '{full_filename2}'")

else:
    print("Графики не сохранены.")
# Отображение графика
plt.show()
print(""""Коэффициент наклона равен 4.5, что означает, что каждый дополнительный час подготовки увеличивает итоговый балл примерно на 4.5 балла.
Коэффициент детерминации R² = 0.99, что указывает на то, что 99% вариации итоговых баллов объясняется количеством часов подготовки. Оставшийся 1% 
изменений могут быть объяснены другими факторами или случайными ошибками. p-значение меньше 0.05, значит, модель статистически значима.  Пик кривой 
находится вблизи нуля, что указывает на то, что среднее значение остатков близко к нулю. Однако, относительно “тяжелые хвосты” и общая сплющенность 
формы свидетельствуют об отклонении от идеального нормального распределения, что однако гипотеза о гетероскедастичности отклонена. Данные не содержат
шум, значит вероятно влияние посторонних факторов.""")


# In[ ]:




