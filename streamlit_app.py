import re
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin

# ----- Определение MetaFeatures (должно совпадать с тем, что было при обучении) -----
class MetaFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if hasattr(X, "values"):
            texts = X.values
        else:
            texts = X
        out = []
        for t in texts:
            out.append([len(t.split()),
                        t.count("!"),
                        t.count("?"),
                        sum(1 for c in t if c.isupper())])
        return np.array(out)
    def get_feature_names_out(self, input_features=None):
        return ["length", "excl", "quest", "upper"]

# Загрузка модели
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "model_meta.joblib"
    if not model_path.exists():
        st.error("Файл model_meta.joblib не найден. Сначала запустите обучение (train.py или ноутбук).")
        st.stop()
    return joblib.load(model_path)

model = load_model()

st.title("Анализ тональности рецензии на фильм")
st.markdown("Введите текст рецензии (минимум 10 символов) — модель предскажет **позитив / негатив / нейтральную** тональность и покажет **5 самых влиятельных слов**.")

review = st.text_area("Текст рецензии", height=200,
                      placeholder="Например: 'This movie was amazing, the acting was brilliant...'")

if st.button("Определить тональность"):
    # ---- Валидация и предупреждения ----
    if not review or len(review.strip()) < 10:
        st.warning("Введите не менее 10 символов.")
        st.stop()

    if not re.search(r"[a-zA-Z]", review):
        st.warning("Текст не содержит английских букв (только цифры, знаки или другие символы). Результат может быть некорректным.")
        st.subheader("Тональность: нейтральная")
        st.metric("Уверенность", "50.00%")
        st.stop()

    if len(review) > 10_000:
        st.warning("Очень длинный текст (> 10 000 символов). Модель может работать медленно, но результат будет получен.")

    # ---- Предсказание ----
    proba = model.predict_proba([review])[0, 1]   # вероятность класса positive

    # Определяем тональность по порогам
    if proba >= 0.6:
        sentiment = "положительная"
    elif proba <= 0.4:
        sentiment = "отрицательная"
    else:
        sentiment = "нейтральная"

    confidence = max(proba, 1 - proba)

    st.subheader(f"Тональность: {sentiment}")
    st.metric("Уверенность", f"{confidence:.2%}")

    # Топ-5 влиятельных слов
    tfidf_vec = model.named_steps["features"].transformer_list[0][1]
    clf = model.named_steps["clf"]
    feature_names = tfidf_vec.get_feature_names_out()
    n_tfidf = len(feature_names)

    tfidf_vector = tfidf_vec.transform([review])
    coefs = clf.coef_[0][:n_tfidf]
    influence = tfidf_vector.toarray()[0] * coefs
    top_indices = np.argsort(influence)[-5:][::-1]

    top_words = [feature_names[i] for i in top_indices]
    df_influence = pd.DataFrame({
        "Слово": top_words,
        "Влияние": influence[top_indices]
    })

    st.subheader("Топ-5 самых влиятельных слов")
    st.dataframe(df_influence, use_container_width=True)

    with st.expander("Мета-признаки этой рецензии"):
        words = review.split()
        st.write(f"* Длина: {len(words)} слов")
        st.write(f"* Восклицательных знаков '!': {review.count('!')}")
        st.write(f"* Вопросительных знаков '?': {review.count('?')}")
        st.write(f"* Заглавных букв: {sum(1 for c in review if c.isupper())}")