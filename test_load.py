import pandas as pd
from datasets import load_dataset

# ... ваш код загрузки ...
dataset = load_dataset("multi_news", split="train", trust_remote_code=True)

# 1. Посмотрим, сколько там данных
print(f"Всего документов: {len(dataset)}")
print(f"Поля данных: {dataset.column_names}") # ['document', 'summary']

# 2. Посмотрим на первый пример
print("\n--- Пример документа (первые 300 символов) ---")
print(dataset[0]['document'][:300]) 
print("\n--- Пример саммари ---")
print(dataset[0]['summary'])

# 3. ЛАЙФХАК ДЛЯ ЛИНЕЙНОЙ АЛГЕБРЫ
# Датасет 'multi_news' огромный (40,000+ статей). 
# Если вы попытаетесь построить матрицу термов на всём датасете, 
# у вас может не хватить оперативной памяти (RAM) или SVD будет считаться вечность.
# Для курсового проекта лучше взять небольшой кусочек (срез).

# Берем первые 100 документов для тестов
mini_dataset = dataset.select(range(100)) 

# Конвертируем в Pandas DataFrame (с таблицами удобнее работать)
df = pd.DataFrame(mini_dataset)

# Сохраним в CSV, чтобы каждый раз не качать заново
df.to_csv("multi_news_subset.csv", index=False)
print("\nСохранил 100 статей в файл multi_news_subset.csv")