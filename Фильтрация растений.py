#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
'''
1. Загрузить или создать DataFrame с разнообразными данными. 
2. Выполнить фильтрацию данных по нескольким условиям. 
3. Сгруппировать данные по одному или нескольким признакам, вычислить агрегированные показатели (среднее, сумма, количество). 
4. Построить визуализации (гистограммы, столбчатые диаграммы, scatter plot) с настройкой параметров (цвет, размер, легенда, подписи осей). 
5. Сохранить результаты анализа и визуализации. 
Рекомендации по выполнению: Используйте методы pandas loc[], 
groupby(), matplotlib/seaborn для построения графиков. Обратите внимание на читаемость и информативность визуализаций.
Спасибо. Теперь давай построим функцию для df = pd.DataFrame(plants_data) 
предназначеннную для вывода гистограмм, столбчатых и точечных диаграмм, с настройкой параметров (цвет, размер, легенда, подписи осей).
'''
# Убираем ограничение по ширине вывода (ширина в символах)
pd.set_option('display.width', 0)  # 0 — означает "без ограничения"
# Убираем перенос длинных ячеек на новую строку (чтобы все было в одной строке)
pd.set_option('display.expand_frame_repr', False)
plants_data = [
    {
        'Растение': 'Шиповник',
        'Категория': 'кустарник',
        'Макс_высота': 3,  # м
        'Макс_кол_стволов': 3, 
        'Особенности': ['шипы', 'привлечение животных'],
        'Условия': 'умеренный климат, солнечные места, плодородная почва'
    },
    {
        'Растение': 'Белладонна',
        'Категория': 'кустарник',
        'Макс_высота': 1.5,
        'Макс_кол_стволов': 1,
        'Особенности': ['яд'],
        'Условия': 'тенистые места, влажная почва'
    },
    {
        'Растение': 'Акация',
        'Категория': 'дерево',
        'Макс_высота': 20,
        'Макс_кол_стволов': 1,
        'Особенности': ['шипы', 'привлечение насекомых', 'крипсис'],
        'Условия': 'сухие почвы, солнечные места'
    },
    {
        'Растение': 'Крапива',
        'Категория': 'трава',
        'Макс_высота': 1.2,
        'Макс_кол_стволов': 1,
        'Особенности': ['шипы', 'яд'],
        'Условия': 'влажные, плодородные почвы'
    },
    {
        'Растение': 'Тополь',
        'Категория': 'дерево',
        'Макс_высота': 30,
        'Макс_кол_стволов': 1,
        'Особенности': ['химическая сигнализация'],
        'Условия': 'влажные почвы, солнечные места'
    },
    {
        'Растение': 'Вереск',
        'Категория': 'кустарничек',
        'Макс_высота': 0.6,
        'Макс_кол_стволов': 2,
        'Особенности': ['привлечение насекомых'],
        'Условия': 'кислые почвы, солнечные места'
    },
    {
        'Растение': 'Орхидея',
        'Категория': 'кустарничек',
        'Макс_высота': 0.3,
        'Макс_кол_стволов': 1,
        'Особенности': ['привлечение насекомых', 'крипсис'],
        'Условия': 'влажный тропический климат, тень'
    },
    {
        'Растение': 'Кактус',
        'Категория': 'кустарник',
        'Макс_высота': 4,
        'Макс_кол_стволов': 3,
        'Особенности': ['шипы'],
        'Условия': 'песчаные, сухие почвы, жаркий климат'
    },
    {
        'Растение': 'Ель',
        'Категория': 'дерево',
        'Макс_высота': 40,
        'Макс_кол_стволов': 1,
        'Особенности': ['шипы', 'химическая сигнализация'],
        'Условия': 'холодный климат, кислые почвы'
    },
    {
        'Растение': 'Малина',
        'Категория': 'кустарник',
        'Макс_высота': 2,
        'Макс_кол_стволов': 5,
        'Особенности': ['шипы', 'привлечение животных'],
        'Условия': 'умеренный климат, солнечные места'
    },
    {
        'Растение': 'Берёза',
        'Категория': 'дерево',
        'Макс_высота': 30,
        'Макс_кол_стволов': 1,
        'Особенности': ['химическая сигнализация'],
        'Условия': 'умеренный климат, солнечные места'
    },
    {
        'Растение': 'Жасмин',
        'Категория': 'кустарник',
        'Макс_высота': 3,
        'Макс_кол_стволов': 3,
        'Особенности': ['привлечение насекомых'],
        'Условия': 'субтропический климат, солнечные места'
    },
    {
        'Растение': 'Лаванда',
        'Категория': 'кустарничек',
        'Макс_высота': 0.8,
        'Макс_кол_стволов': 2,
        'Особенности': ['привлечение насекомых'],
        'Условия': 'сухие, солнечные места'
    },
    {
        'Растение': 'Одуванчик',
        'Категория': 'трава',
        'Макс_высота': 0.4,
        'Макс_кол_стволов': 1,
        'Особенности': ['привлечение насекомых'],
        'Условия': 'различные почвы, солнечные места'
    },
    {
        'Растение': 'Клён',
        'Категория': 'дерево',
        'Макс_высота': 35,
        'Макс_кол_стволов': 1,
        'Особенности': ['химическая сигнализация'],
        'Условия': 'умеренный климат, плодородные почвы'
    },
    {
        'Растение': 'Шалфей',
        'Категория': 'кустарничек',
        'Макс_высота': 0.6,
        'Макс_кол_стволов': 2,
        'Особенности': ['привлечение насекомых'],
        'Условия': 'сухие, солнечные места'
    },
    {
        'Растение': 'Можжевельник',
        'Категория': 'кустарник',
        'Макс_высота': 5,
        'Макс_кол_стволов': 3,
        'Особенности': ['шипы'],
        'Условия': 'сухие почвы, солнечные места'
    },
    {
        'Растение': 'Папоротник',
        'Категория': 'трава',
        'Макс_высота': 1,
        'Макс_кол_стволов': 1,
        'Особенности': [],
        'Условия': 'влажные, тенистые места'
    },
    {
        'Растение': 'Роза',
        'Категория': 'кустарник',
        'Макс_высота': 2,
        'Макс_кол_стволов': 4,
        'Особенности': ['шипы', 'привлечение насекомых', 'привлечение животных'],
        'Условия': 'умеренный климат, солнечные места'
    },
    {
        'Растение': 'Сосна',
        'Категория': 'дерево',
        'Макс_высота': 40,
        'Макс_кол_стволов': 1,
        'Особенности': ['химическая сигнализация', 'шипы'],
        'Условия': 'сухие, песчаные почвы, солнечные места'
    },
    {
        'Растение': 'Клен ясенелистный',
        'Категория': 'дерево',
        'Макс_высота': 25,
        'Макс_кол_стволов': 1,
        'Особенности': ['химическая сигнализация'],
        'Условия': 'умеренный климат, плодородные почвы'
    },
    {
        'Растение': 'Черника',
        'Категория': 'кустарничек',
        'Макс_высота': 0.6,
        'Макс_кол_стволов': 2,
        'Особенности': ['привлечение животных'],
        'Условия': 'кислые почвы, тенистые места'
    },
    {
        'Растение': 'Пихта',
        'Категория': 'дерево',
        'Макс_высота': 40,
        'Макс_кол_стволов': 1,
        'Особенности': ['химическая сигнализация', 'шипы'],
        'Условия': 'холодный климат, влажные почвы'
    },
    {
        'Растение': 'Клюква',
        'Категория': 'кустарничек',
        'Макс_высота': 0.3,
        'Макс_кол_стволов': 1,
        'Особенности': ['привлечение животных'],
        'Условия': 'кислые, влажные почвы'
    },
    {
        'Растение': 'Ива',
        'Категория': 'кустарник',
        'Макс_высота': 10,
        'Макс_кол_стволов': 3,
        'Особенности': ['химическая сигнализация'],
        'Условия': 'влажные места, солнечные и полутенистые'
    },
    {
        'Растение': 'Фиалка',
        'Категория': 'трава',
        'Макс_высота': 0.2,
        'Макс_кол_стволов': 1,
        'Особенности': ['привлечение насекомых'],
        'Условия': 'влажные, тенистые места'
    },
    {
        'Растение': 'Кипарис',
        'Категория': 'дерево',
        'Макс_высота': 30,
        'Макс_кол_стволов': 1,
        'Особенности': ['химическая сигнализация'],
        'Условия': 'субтропический климат, солнечные места'
    },
    {
        'Растение': 'Мята',
        'Категория': 'трава',
        'Макс_высота': 0.9,
        'Макс_кол_стволов': 1,
        'Особенности': ['привлечение насекомых'],
        'Условия': 'влажные почвы, солнечные и полутенистые места'
    },
    {
        'Растение': 'Барбарис',
        'Категория': 'кустарник',
        'Макс_высота': 3,
        'Макс_кол_стволов': 3,
        'Особенности': ['шипы', 'привлечение животных'],
        'Условия': 'сухие, солнечные места'
    },
    {
        'Растение': 'Гортензия',
        'Категория': 'кустарник',
        'Макс_высота': 2,
        'Макс_кол_стволов': 2,
        'Особенности': ['привлечение насекомых'],
        'Условия': 'влажные, тенистые места'
    }
    # Добавьте остальные растения аналогично с реальными данными
]

df = pd.DataFrame(plants_data)
def plot_plants_height(df):
    required_cols = ['Растение', 'Макс_высота', 'Категория']
    for col in required_cols:
        if col not in df.columns:
            print(f"В DataFrame отсутствует столбец '{col}'")
            return

    # Сортируем по высоте
    df_sorted = df.sort_values('Макс_высота')

    # Цвета для категорий
    color_map = {
        'дерево': 'brown',
        'трава': 'cornflowerblue',
        'кустарник': 'springgreen',
        'куст': 'rosybrown'
    }

    # Определяем цвет для каждой категории (с приведением к нижнему регистру)
    colors = [color_map.get(cat.lower(), 'gray') for cat in df_sorted['Категория']]

    plt.figure(figsize=(max(10, len(df_sorted)*0.5), 6))  # ширина зависит от числа растений

    bars = plt.bar(range(len(df_sorted)), df_sorted['Макс_высота'], color=colors)

    # Подписи названий растений под столбцами, повернуты для читаемости
    plt.xticks(range(len(df_sorted)), df_sorted['Растение'], rotation=45, ha='right')

    plt.xlabel('Растение')
    plt.ylabel('Максимальная высота')
    plt.title('Высота растений (от низкой к высокой)')

    plt.tight_layout()
    plt.show()
plot_plants_height(df)  # передайте отфильтрованный DataFrame
def plot_category_counts(df):
    # Категории и цвета
    categories = ['дерево', 'трава', 'кустарник', 'кустарничек']
    color_map = {
        'дерево': 'brown',
        'трава': 'cornflowerblue',
        'кустарничек': 'springgreen',
        'кустарник': 'rosybrown'
    }

    # Считаем количество растений по категориям (приводим к нижнему регистру для учета регистра)
    counts = []
    for cat in categories:
        count = df['Категория'].str.lower().value_counts().get(cat, 0)
        counts.append(count)

    plt.figure(figsize=(8, 5))
    bars = plt.bar(categories, counts, color=[color_map[cat] for cat in categories])

    # Подписи количества сверху над столбцами
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.1, str(count), ha='center', va='bottom')

    plt.xlabel('Категория')
    plt.ylabel('Количество растений')
    plt.title('Количество растений по категориям')

    plt.tight_layout()
    plt.show()
plot_category_counts(df)

def plot_conditions_bands_split(df, n_parts=3):
    # Проверка столбцов
    for col in ['Растение', 'Категория', 'Условия']:
        if col not in df.columns:
            print(f"В DataFrame отсутствует столбец '{col}'")
            return

    def parse_conditions(x):
        if isinstance(x, list):
            return [cond.strip() for cond in x if cond.strip()]
        elif isinstance(x, str):
            return [cond.strip() for cond in x.split(',') if cond.strip()]
        else:
            return []

    df = df.copy()
    df['Список_условий'] = df['Условия'].apply(parse_conditions)

    all_conditions = []
    for conds in df['Список_условий']:
        for cond in conds:
            if cond not in all_conditions:
                all_conditions.append(cond)

    color_map = {
        'дерево': 'brown',
        'трава': 'cornflowerblue',
        'кустарничек': 'springgreen',
        'кустарник': 'rosybrown'
    }

    total_conditions = len(all_conditions)
    part_size = math.ceil(total_conditions / n_parts)

    for part_idx in range(n_parts):
        start_idx = part_idx * part_size
        end_idx = min(start_idx + part_size, total_conditions)
        conditions_part = all_conditions[start_idx:end_idx]

        n_conditions = len(conditions_part)
        if n_conditions == 0:
            continue

        fig_width = max(12, n_conditions * 1.5)
        fig_height = 8
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        x_positions = np.arange(n_conditions) * 2

        for x in x_positions:
            ax.axvspan(x - 0.75, x + 0.75, color='lightgray', alpha=0.3)

        labels_dict = {}

        for i, condition in enumerate(conditions_part):
            subset = df[df['Список_условий'].apply(lambda conds: condition in conds)].reset_index(drop=True)
            colors = [color_map.get(cat.lower(), 'gray') for cat in subset['Категория']]
            numbers = range(1, len(subset) + 1)

            ax.scatter([x_positions[i]]*len(subset), numbers, c=colors, s=120, edgecolors='black', zorder=3)

            labels_dict[condition] = list(zip(numbers, subset['Растение']))

        ax.set_xticks(x_positions)
        ax.set_xticklabels(conditions_part, rotation=65, ha='right', fontsize=10)
        ax.set_xlim(x_positions[0] - 1, x_positions[-1] + 1)

        max_plants = max(len(plants) for plants in labels_dict.values()) if labels_dict else 1
        ax.set_ylim(0, max_plants + 1)
        ax.set_ylabel('Номер растения внутри пояса', fontsize=12)
        ax.set_title(f'Распределение растений по условиям (поясам) — часть {part_idx + 1} из {n_parts}', fontsize=14)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=cat,
                   markerfacecolor=color, markersize=12, markeredgecolor='black')
            for cat, color in color_map.items()
        ]
        ax.legend(handles=legend_elements, title='Категория', loc='upper right', fontsize=11, title_fontsize=12)

        plt.tight_layout()
        plt.show()

        print(f"Расшифровка номеров растений по условиям — часть {part_idx + 1} из {n_parts}:")
        for condition in conditions_part:
            print(f"\nУсловие: {condition}")
            for num, name in labels_dict[condition]:
                print(f"  {num}: {name}")
plot_conditions_bands_split(df, n_parts=3)
def filter_plants(df):
    print("Введите условия фильтрации через запятую.")
    print("Доступные поля для фильтрации: Категория, Особенности, Условия")
    user_input = input("Пример: кустарник, шипы, солнечные места\nВаш ввод: ")

    filters = [f.strip().lower() for f in user_input.split(',') if f.strip()]
    if not filters:
        print("Условия не введены, возвращаю полный список.")
        return df

    # Уникальные категории и особенности для проверки
    all_categories = set(df['Категория'].str.lower())
    all_features = set()
    for feats in df['Особенности']:
        all_features.update([f.lower() for f in feats])

    # Выделяем фильтры по категориям, особенностям и условиям
    requested_categories = set(f for f in filters if f in all_categories)
    requested_features = set(f for f in filters if f in all_features)
    # Остальные фильтры считаем условиями произрастания (те, что не категория и не особенность)
    requested_conditions = set(f for f in filters if f not in requested_categories and f not in requested_features)

    filtered_df = df

    # 1. Фильтрация по категориям (если есть)
    if requested_categories:
        filtered_df = filtered_df[filtered_df['Категория'].str.lower().isin(requested_categories)]
        if filtered_df.empty:
            print("Нет растений с указанной категорией.")
            return filtered_df

    # 2. Фильтрация по особенностям (если есть)
    if requested_features:
        def has_feature(feats):
            feats_lower = [f.lower() for f in feats]
            # Растение должно иметь хотя бы одну из запрошенных особенностей
            return any(f in feats_lower for f in requested_features)
        filtered_df = filtered_df[filtered_df['Особенности'].apply(has_feature)]
        if filtered_df.empty:
            print("Нет растений с указанными особенностями.")
            return filtered_df

    # 3. Фильтрация по условиям произрастания (если есть)
    if requested_conditions:
        def has_condition(cond):
            cond_lower = cond.lower()
            return any(c in cond_lower for c in requested_conditions)
        filtered_df = filtered_df[filtered_df['Условия'].apply(has_condition)]
        if filtered_df.empty:
            print("Нет растений с указанными условиями произрастания.")
            return filtered_df

    print(f"Найдено растений: {len(filtered_df)}")
    return filtered_df
def average_stems(filtered_df):
    if filtered_df.empty:
        print("Нет растений для расчёта среднего числа стволов.")
        return None
    # Замените 'Макс_кол_стволов' на название столбца с числом стволов в вашем df
    avg = filtered_df['Макс_кол_стволов'].mean()
    print(f"Среднее число стволов: {avg:.2f}")
    return avg


# Использование:
df_filtered = filter_plants(df)
if not df_filtered.empty:
    average_stems(df_filtered)
print(df_filtered)
def save_to_excel(df, filename='results.xlsx'):
    try:
        df.to_excel(filename, index=False)
        print(f"Результаты сохранены в файл: {filename}")
    except Exception as e:
        print(f"Ошибка при сохранении файла: {e}")


# Основной блок работы
if not df_filtered.empty:
    average_stems(df_filtered)
    save_answer = input("Сохранить результаты в Excel? (да/нет): ").strip().lower()
    if save_answer in ['да', 'д', 'yes', 'y']:
        filename = input("Введите имя файла для сохранения (например, results.xlsx): ").strip()
        if not filename:
            filename = 'results.xlsx'
        save_to_excel(df_filtered, filename)
else:
    print("Отфильтрованный список пуст, ничего не сохраняем.")


# In[ ]:




