#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# 1. Web Scraping
def scrape_text(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.content, 'html.parser')
        text_container = soup.find('div', class_='text')
        if not text_container:
            print("Ошибка: <div class='text'> не найден.")
            return None
        paragraphs = text_container.find_all('p')
        if not paragraphs:
            print("Ошибка: <p> теги не найдены.")
            return None
        text = '\n'.join([p.text for p in paragraphs])
        return text
    except requests.exceptions.RequestException as e:
        print(f"Запрос {url} не прошёл: {e}")
        return None
    except Exception as e:
        print(f"Ошибка прочтения: {e}")
        return None

url = "https://studopedia.ru/19_48539_syuzhetno-kompozitsionnaya-i-slovesnaya-organizatsiya-dramaticheskogo-teksta.html"
raw_text = scrape_text(url)
if not raw_text:
    print("Неудалось прочесть ссылку. Закрытие программы...")
    exit()

# 2. Очищение текста
def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    tokens = word_tokenize(text)
    return tokens

tokens = clean_text(raw_text)
# 3. Создание DataFrame и базовый анализ
def analyze_text(tokens):
    df = pd.DataFrame({'word': tokens})
    word_counts = df['word'].value_counts().reset_index()
    word_counts.columns = ['word', 'frequency']
    stop_words = set(stopwords.words('russian'))
    filtered_df = word_counts[~word_counts['word'].isin(stop_words)]

    print("\nПервые три предложение до удаления слов:")
    sentences_before = nltk.sent_tokenize(raw_text)
    for i in range(min(3, len(sentences_before))):
        print(f"Sentence {i+1}: {sentences_before[i]}")

    print("\nПервые три предложение после удаления слов:")

    # Токенизация и удаление
    sentences_after = []
    for sentence in nltk.sent_tokenize(raw_text):
        tokens_in_sentence = clean_text(sentence)
        filtered_tokens = [word for word in tokens_in_sentence if word not in stop_words]
        sentences_after.append(" ".join(filtered_tokens))

    for i in range(min(3, len(sentences_after))):
        print(f"Sentence {i+1}: {sentences_after[i]}")

    return filtered_df

filtered_df = analyze_text(tokens)

print("\n20 слов после удаления:")
print(filtered_df.head(20))
# 4. Подготовка к анализу
# Функция для фильтрации слов по длине 
filtered_df['is_long'] = filtered_df['word'].apply(lambda x: len(x) >= 6)


print("\nФильтрация слов по длине:")
print(filtered_df.head(20))
# Можно сохранить в csv файл
# filtered_df.to_csv('word_frequencies.csv', index=False)
#Советы от нейросетки для будущих самостоятельных работ
#  Further steps for actual modeling would include:
#  - Converting categorical features (like 'is_long') to numerical (e.g., one-hot encoding).
#  - Potentially using TF-IDF vectorization or word embeddings (word2vec, GloVe, etc.) to represent the words numerically.
#  - Splitting the data into training and testing sets if you are building a model.
#  - Choosing and training a suitable NLP model (e.g., sentiment analysis, text classification, topic modeling).


# In[ ]:




