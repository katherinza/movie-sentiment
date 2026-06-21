import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_positive_review():
    response = client.post("/predict", json={"review": "This is a fantastic and brilliant movie!"})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "positive"
    assert 0.5 <= data["confidence"] <= 1.0

def test_negative_review():
    response = client.post("/predict", json={"review": "That was the worst film I have ever seen."})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "negative"
    assert 0.0 <= data["confidence"] <= 0.5

def test_empty_text():
    response = client.post("/predict", json={"review": ""})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "neutral"
    assert data["confidence"] == 0.5
    assert data["warning"] is not None

def test_short_text():
    response = client.post("/predict", json={"review": "Bad"})
    assert response.status_code == 200
    data = response.json()
    # Ожидаем, что предупреждение о коротком тексте есть
    assert data["warning"] is not None
    # Сама модель всё равно даст какой-то ответ (positive/negative)
    assert data["sentiment"] in ("positive", "negative", "neutral")

def test_only_punctuation():
    response = client.post("/predict", json={"review": "!!!!????...!!!"})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "neutral"
    assert data["warning"] is not None

def test_long_text():
    long_review = "Very good movie. " * 500
    response = client.post("/predict", json={"review": long_review})
    assert response.status_code == 200
    assert "sentiment" in response.json()

def test_missing_field():
    response = client.post("/predict", json={})
    assert response.status_code == 422   # Pydantic validation error

def test_confidence_range():
    response = client.post("/predict", json={"review": "A nice little indie film with heart."})
    assert response.status_code == 200
    conf = response.json()["confidence"]
    assert 0.0 <= conf <= 1.0

def test_warning_on_too_long():
    # Отправляем текст длиннее 10 000 символов
    long_review = "a" * 10001
    response = client.post("/predict", json={"review": long_review})
    assert response.status_code == 200
    data = response.json()
    assert data["warning"] is not None
    assert "длин" in data["warning"].lower() or "10" in data["warning"]