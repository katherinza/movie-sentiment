import re
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin

# ----- Определение класса MetaFeatures (должно совпадать с обучением) -----

# Класс для генерации мета-признаков
class MetaFeatures(BaseEstimator, TransformerMixin):
    # Метод fit (без изменений)
    def fit(self, X, y=None):
        return self
    # Метод transform для извлечения признаков
    def transform(self, X):
        # Преобразование входных данных в список текстов
        if hasattr(X, "values"):
            texts = X.values
        else:
            texts = X
        # Список признаков для каждого текста
        out = []
        # Цикл по текстам
        for t in texts:
            out.append([len(t.split()),      # длина в словах
                        t.count("!"),        # восклицательные
                        t.count("?"),        # вопросительные
                        sum(1 for c in t if c.isupper())])  # заглавные буквы
        return np.array(out)
    # Метод для имён признаков
    def get_feature_names_out(self, input_features=None):
        return ["length", "excl", "quest", "upper"]

# ----- Загрузка модели -----

# Функция загрузки модели с кешированием
@st.cache_resource
def load_model():
    # Путь к файлу модели
    model_path = Path(__file__).parent / "model_meta.joblib"
    # Проверка существования файла
    if not model_path.exists():
        st.error("Файл model_meta.joblib не найден. Сначала запустите обучение (train.py или ноутбук).")
        st.stop()
    # Загрузка модели из файла
    return joblib.load(model_path)

# Вызов загрузки модели
model = load_model()

# ----- Интерфейс Streamlit -----

# Заголовок приложения
st.title("Анализ тональности рецензии на фильм")
# Пояснительный текст
st.markdown("Введите текст рецензии (минимум 10 символов) — модель предскажет **позитив / негатив / нейтральную** тональность и покажет **5 самых влиятельных слов**.")

# Поле ввода текста
review = st.text_area("Текст рецензии", height=200,
                      placeholder="Например: 'This movie was amazing, the acting was brilliant...'")

# Кнопка для запуска анализа
if st.button("Определить тональность"):
    # ---- Валидация входного текста ----

    # Проверка минимальной длины
    if not review or len(review.strip()) < 10:
        st.warning("Введите не менее 10 символов.")
        st.stop()

    # Проверка наличия английских букв
    if not re.search(r"[a-zA-Z]", review):
        st.warning("Текст не содержит английских букв (только цифры, знаки или другие символы). Результат может быть некорректным.")
        st.subheader("Тональность: нейтральная")
        st.metric("Уверенность", "50.00%")
        st.stop()

    # Предупреждение о длинном тексте
    if len(review) > 10_000:
        st.warning("Очень длинный текст (> 10 000 символов). Модель может работать медленно, но результат будет получен.")

    # ---- Предсказание тональности ----

    # Получение вероятности положительного класса
    proba = model.predict_proba([review])[0, 1]

    # Определение тональности по порогам
    if proba >= 0.6:
        sentiment = "положительная"
    elif proba <= 0.4:
        sentiment = "отрицательная"
    else:
        sentiment = "нейтральная"

    # Вычисление уверенности
    confidence = max(proba, 1 - proba)

    # Вывод результата
    st.subheader(f"Тональность: {sentiment}")
    st.metric("Уверенность", f"{confidence:.2%}")

    # ---- Анализ влиятельных слов ----

    # Извлечение TF-IDF векторизатора из пайплайна
    tfidf_vec = model.named_steps["features"].transformer_list[0][1]
    # Извлечение классификатора
    clf = model.named_steps["clf"]
    # Получение имён признаков
    feature_names = tfidf_vec.get_feature_names_out()
    # Количество TF-IDF признаков
    n_tfidf = len(feature_names)

    # Преобразование текста в TF-IDF вектор
    tfidf_vector = tfidf_vec.transform([review])
    # Коэффициенты модели для TF-IDF признаков
    coefs = clf.coef_[0][:n_tfidf]
    # Вычисление влияния каждого слова (произведение TF-IDF на коэффициент)
    influence = tfidf_vector.toarray()[0] * coefs
    # Индексы пяти слов с наибольшим влиянием (по убыванию)
    top_indices = np.argsort(influence)[-5:][::-1]

    # Извлечение самих слов
    top_words = [feature_names[i] for i in top_indices]
    # Создание DataFrame для вывода
    df_influence = pd.DataFrame({
        "Слово": top_words,
        "Влияние": influence[top_indices]
    })

    # Вывод таблицы влиятельных слов
    st.subheader("Топ-5 самых влиятельных слов")
    st.dataframe(df_influence, use_container_width=True)

    # ---- Дополнительная информация: мета-признаки ----

    # Раскрывающийся блок с мета-признаками
    with st.expander("Мета-признаки этой рецензии"):
        # Количество слов
        words = review.split()
        st.write(f"* Длина: {len(words)} слов")
        st.write(f"* Восклицательных знаков '!': {review.count('!')}")
        st.write(f"* Вопросительных знаков '?': {review.count('?')}")
        st.write(f"* Заглавных букв: {sum(1 for c in review if c.isupper())}")