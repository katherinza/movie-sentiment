import re
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
# Импорт библиотеки для перевода текста
from deep_translator import GoogleTranslator

# ----------------------- Стилизация интерфейса -----------------------
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    .result-card {
        background: none;
        padding: 1rem 0;
        margin: 1rem 0;
        text-align: left;
        box-shadow: none;
    }
    .positive { color: #198754; font-weight: 700; }
    .negative { color: #dc3545; font-weight: 700; }
    .neutral  { color: #6c757d; font-weight: 700; }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# ----------------------- Класс генерации мета-признаков -----------------------
class MetaFeatures(BaseEstimator, TransformerMixin):
    # Метод обучения (без изменений)
    def fit(self, X, y=None):
        return self
    # Метод трансформации текста в мета-признаки
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
    # Метод получения имён признаков
    def get_feature_names_out(self, input_features=None):
        return ["length", "excl", "quest", "upper"]

# ----------------------- Загрузка модели с кешированием -----------------------
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "model_meta.joblib"
    if not model_path.exists():
        st.error("Файл model_meta.joblib не найден. Сначала запустите обучение.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# ----------------------- Функции для работы с кириллицей и переводом -----------------------
# Функция проверки наличия кириллических символов
def contains_cyrillic(text):
    return bool(re.search(r'[а-яёА-ЯЁ]', text))

# Функция перевода текста на английский язык
def translate_to_english(text):
    try:
        translator = GoogleTranslator(source='auto', target='en')
        return translator.translate(text)
    except Exception as e:
        st.warning(f"Не удалось перевести текст: {e}. Использую оригинал.")
        return text

# ----------------------- Боковая панель -----------------------
with st.sidebar:
    st.markdown("## О проекте")
    st.markdown("""
    **ML-03: Тональность рецензий**  
    Модель обучена на 50 000 отзывов IMDb.
    
    **Метрики лучшей модели:**  
    - Accuracy: 0.90  
    - F1-macro: 0.90  
    - ROC-AUC: 0.96
    
    **Пороги тональности:**  
    >= 0.6 → позитивная  
    <= 0.4 → негативная  
    иначе → нейтральная
    """)
    st.divider()
    st.markdown("### Примеры рецензий")
    # Кнопки для заполнения поля русскоязычными примерами
    if st.button("Позитивный пример"):
        st.session_state.review = "Этот фильм просто великолепен! Актёрская игра на высоте, сюжет держит в напряжении до самого конца."
    if st.button("Негативный пример"):
        st.session_state.review = "Ужасное кино, пустая трата времени. Сценарий отвратительный, игра актёров деревянная."
    if st.button("Смешанный пример"):
        st.session_state.review = "Спецэффекты впечатляют, но сюжет слабоват и концовка разочаровала. Не уверен, что стал бы советовать."

# ----------------------- Главная страница -----------------------
# Заголовок и подзаголовок
st.markdown('<div class="title">Анализ тональности рецензий</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Введите рецензию (минимум 10 символов) — модель определит позитив, негатив или нейтральную окраску и покажет 5 самых влиятельных слов.</div>', unsafe_allow_html=True)

# Поле ввода текста
default_text = st.session_state.get("review", "")
review = st.text_area("Текст рецензии (русский/английский)", value=default_text, height=180,
                      placeholder="Например: 'Этот фильм просто великолепен...' или 'Terrible movie...'")

# Кнопка запуска анализа
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button("Определить тональность", use_container_width=True)

# ----------------------- Обработка нажатия кнопки -----------------------
if analyze_button:
    # ---- Валидация текста ----
    # Проверка минимальной длины
    if not review or len(review.strip()) < 10:
        st.warning("Введите не менее 10 символов.")
        st.stop()

    # Проверка языка текста и перевод (если кириллица)
    original_review = review
    if contains_cyrillic(review):
        with st.spinner("Перевод на английский..."):
            review = translate_to_english(review)
            if review != original_review:
                st.info(f"Перевод: {review}")

    # Проверка наличия английских букв (после возможного перевода)
    if not re.search(r"[a-zA-Z]", review):
        st.warning("Текст не содержит английских букв — результат нейтральный.")
        st.markdown('<div class="result-card"><h3 style="text-align:left;" class="neutral">Нейтральная</h3>'
                    '<p style="color:#6c757d;">Уверенность 50%</p></div>',
                    unsafe_allow_html=True)
        st.stop()

    # Предупреждение о слишком длинном тексте
    if len(review) > 10_000:
        st.warning("Текст очень длинный (>10 000 символов), анализ может занять несколько секунд.")

    # ---- Предсказание тональности ----
    proba = model.predict_proba([review])[0, 1]

    if proba >= 0.6:
        sentiment_class = "positive"
        sentiment_text = "Положительная"
    elif proba <= 0.4:
        sentiment_class = "negative"
        sentiment_text = "Отрицательная"
    else:
        sentiment_class = "neutral"
        sentiment_text = "Нейтральная"

    confidence = max(proba, 1 - proba)

    # ---- Вывод результата ----
    st.markdown(f"""
    <div class="result-card">
        <h2 style="text-align:left;">Тональность: <span class="{sentiment_class}">{sentiment_text}</span></h2>
        <p style="font-size:1.2rem; color:#495057; text-align:left;">Уверенность: <strong>{confidence:.2%}</strong></p>
        <div style="max-width:400px;">
    """, unsafe_allow_html=True)

    st.progress(confidence)
    st.markdown("</div></div>", unsafe_allow_html=True)

    # ---- Топ-5 влиятельных слов ----
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

    # ---- Построение столбчатой диаграммы с Plotly ----
    try:
        import plotly.graph_objects as go

        # Функция переноса длинных слов
        def wrap_word(word, max_chars=8):
            return '\n'.join([word[i:i+max_chars] for i in range(0, len(word), max_chars)])

        wrapped_words = [wrap_word(w) for w in top_words]

        # Выбор цвета столбцов в зависимости от общей тональности
        if sentiment_class == "positive":
            bar_colors = ['#198754' if v > 0 else '#dc3545' for v in influence[top_indices]]
        elif sentiment_class == "negative":
            bar_colors = ['#dc3545'] * len(top_indices)
        else:  # neutral
            bar_colors = ['#6c757d'] * len(top_indices)

        fig = go.Figure(data=[go.Bar(
            x=wrapped_words,
            y=influence[top_indices],
            marker_color=bar_colors,
            text=[f'{v:.3f}' for v in influence[top_indices]],
            textposition='outside'
        )])
        fig.update_layout(
            xaxis=dict(
                tickangle=0,
                tickfont=dict(size=12),
            ),
            yaxis_title="Влияние",
            margin=dict(b=80),
            template='plotly_white',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.warning("Для улучшенной визуализации установите plotly: `pip install plotly`. Пока используется стандартный график.")
        st.bar_chart(df_influence.set_index("Слово"), use_container_width=True)

    # ---- Отображение таблицы с цветовой кодировкой ----
    def make_color_func(sent):
        def color_influence(val):
            if sent == "positive":
                color = 'green' if val > 0 else 'red'
            elif sent == "negative":
                color = 'red'
            else:  # neutral
                color = '#6c757d'
            return f'color: {color}; font-weight: bold;'
        return color_influence

    styled_df = df_influence.style.applymap(make_color_func(sentiment_class), subset=["Влияние"])
    st.dataframe(styled_df, use_container_width=True)

    # ---- Мета-признаки (раскрывающийся блок) ----
    with st.expander("Мета-признаки этой рецензии"):
        words = review.split()
        col_a, col_b = st.columns(2)
        col_a.metric("Длина (слов)", len(words))
        col_a.metric("Восклицательные знаки '!'", review.count("!"))
        col_b.metric("Вопросительные знаки '?'", review.count("?"))
        col_b.metric("Заглавные буквы", sum(1 for c in review if c.isupper()))