import re
import joblib
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import __main__
from deep_translator import GoogleTranslator

# ---------- Класс генерации мета-признаков ----------
class MetaFeatures(BaseEstimator, TransformerMixin):
    # Метод обучения (без изменений)
    def fit(self, X, y=None):
        return self
    # Метод трансформации текста в признаки
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
    # Метод имён признаков
    def get_feature_names_out(self, input_features=None):
        return ["length", "excl", "quest", "upper"]

# ---------- Регистрация класса в __main__ для корректной загрузки pickle ----------
__main__.MetaFeatures = MetaFeatures

# ---------- Загрузка модели ----------
MODEL_PATH = Path(__file__).resolve().parent.parent / "model_meta.joblib"

model = None
if MODEL_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Модель загружена из {MODEL_PATH}")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
else:
    print(f"Файл модели не найден: {MODEL_PATH}")

# ---------- Функции перевода и валидации ----------
# Функция проверки наличия кириллицы
def contains_cyrillic(text: str) -> bool:
    return bool(re.search(r'[а-яёА-ЯЁ]', text))

# Функция перевода текста на английский
def translate_to_english(text: str) -> str:
    try:
        translator = GoogleTranslator(source='auto', target='en')
        return translator.translate(text)
    except Exception as e:
        print(f"Ошибка перевода: {e}")
        return text

# Функция валидации текста (возвращает текст предупреждения или None)
def validate_review(text: str, allow_russian: bool = True) -> Optional[str]:
    if not text or not text.strip():
        return "Текст пуст или состоит только из пробелов."
    if len(text.strip()) < 10:
        return "Слишком короткий текст (менее 10 символов)."
    if allow_russian:
        if not re.search(r"[a-zA-Zа-яёА-ЯЁ]", text):
            return "Текст не содержит букв."
    else:
        if not re.search(r"[a-zA-Z]", text):
            return "Текст не содержит английских букв."
    if len(text) > 10_000:
        return "Очень длинный текст (более 10 000 символов)."
    return None

# ---------- Настройка FastAPI-приложения ----------
app = FastAPI(title="Movie Sentiment API")

# ---------- Модели данных ----------
class ReviewRequest(BaseModel):
    review: str

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    warning: Optional[str] = None

# ---------- Эндпоинт проверки работоспособности ----------
@app.get("/health")
async def health():
    return {"status": "ok"}

# ---------- Эндпоинт предсказания тональности ----------
@app.post("/predict", response_model=SentimentResponse)
async def predict(data: ReviewRequest):
    # Проверка наличия загруженной модели
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    # Валидация текста (с учётом русского языка)
    warning = validate_review(data.review, allow_russian=True)

    original_text = data.review
    text_for_model = original_text

    # Проверка наличия кириллицы и перевод при необходимости
    if contains_cyrillic(original_text):
        translated = translate_to_english(original_text)
        if translated == original_text:
            warning = (warning + " Не удалось перевести русский текст.") if warning else "Не удалось перевести русский текст."
        else:
            text_for_model = translated

    # Проверка наличия английских букв после возможного перевода
    if not re.search(r"[a-zA-Z]", text_for_model):
        return SentimentResponse(sentiment="neutral", confidence=0.5, warning=warning)

    # Проверка на пустой текст после обработки
    if not text_for_model.strip():
        return SentimentResponse(sentiment="neutral", confidence=0.5,
                                warning="Текст после обработки оказался пустым.")

    # Предсказание вероятности и определение тональности
    proba = model.predict_proba([text_for_model])[0, 1]
    sentiment = "positive" if proba >= 0.5 else "negative"
    return SentimentResponse(sentiment=sentiment, confidence=round(float(proba), 4), warning=warning)