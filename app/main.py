import re
import joblib
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import sys
import __main__

# ---------- Определение MetaFeatures ----------
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

# ---------- Регистрируем класс в __main__ (где его ищет pickle) ----------
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

# ---------- FastAPI ----------
app = FastAPI(title="Movie Sentiment API")

class ReviewRequest(BaseModel):
    review: str

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    warning: Optional[str] = None

def validate_review(text: str) -> Optional[str]:
    if not text or not text.strip():
        return "Текст пуст или состоит только из пробелов."
    if len(text.strip()) < 10:
        return "Слишком короткий текст (менее 10 символов)."
    if not re.search(r"[a-zA-Z]", text):
        return "Текст не содержит букв."
    if len(text) > 10_000:
        return "Очень длинный текст (более 10 000 символов)."
    return None

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict", response_model=SentimentResponse)
async def predict(data: ReviewRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    warning = validate_review(data.review)
    text = data.review.lower()

    if not text.strip() or not re.search(r"[a-zA-Z]", text):
        return SentimentResponse(sentiment="neutral", confidence=0.5, warning=warning)

    proba = model.predict_proba([data.review])[0, 1]
    sentiment = "positive" if proba >= 0.5 else "negative"
    return SentimentResponse(sentiment=sentiment, confidence=round(float(proba), 4), warning=warning)