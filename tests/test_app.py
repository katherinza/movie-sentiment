# Добавление корневой директории проекта в путь импорта
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Импорт тестового клиента и FastAPI-приложения
from fastapi.testclient import TestClient
from app.main import app

# Создание тестового клиента для отправки HTTP-запросов
client = TestClient(app)

# Тест эндпоинта /health
def test_health():
    # GET-запрос к /health
    response = client.get("/health")
    # Проверка статуса ответа
    assert response.status_code == 200
    # Проверка содержимого JSON-ответа
    assert response.json() == {"status": "ok"}

# Тест предсказания для положительного отзыва
def test_positive_review():
    # POST-запрос с положительным текстом
    response = client.post("/predict", json={"review": "This is a fantastic and brilliant movie!"})
    # Проверка статуса 200
    assert response.status_code == 200
    # Извлечение данных ответа
    data = response.json()
    # Проверка тональности
    assert data["sentiment"] == "positive"
    # Проверка диапазона уверенности
    assert 0.5 <= data["confidence"] <= 1.0

# Тест предсказания для отрицательного отзыва
def test_negative_review():
    # POST-запрос с отрицательным текстом
    response = client.post("/predict", json={"review": "That was the worst film I have ever seen."})
    # Статус ответа
    assert response.status_code == 200
    # Данные ответа
    data = response.json()
    # Тональность
    assert data["sentiment"] == "negative"
    # Уверенность (для negative обычно <0.5)
    assert 0.0 <= data["confidence"] <= 0.5

# Тест обработки пустого текста
def test_empty_text():
    # POST-запрос с пустой строкой
    response = client.post("/predict", json={"review": ""})
    # Статус
    assert response.status_code == 200
    # Данные
    data = response.json()
    # Тональность по умолчанию
    assert data["sentiment"] == "neutral"
    # Уверенность 0.5
    assert data["confidence"] == 0.5
    # Наличие предупреждения
    assert data["warning"] is not None

# Тест обработки слишком короткого текста
def test_short_text():
    # POST-запрос с коротким словом
    response = client.post("/predict", json={"review": "Bad"})
    # Статус
    assert response.status_code == 200
    # Данные
    data = response.json()
    # Предупреждение о коротком тексте
    assert data["warning"] is not None
    # Модель всё равно возвращает какую-то тональность
    assert data["sentiment"] in ("positive", "negative", "neutral")

# Тест текста, состоящего только из пунктуации
def test_only_punctuation():
    # POST-запрос со знаками препинания
    response = client.post("/predict", json={"review": "!!!!????...!!!"})
    # Статус
    assert response.status_code == 200
    # Данные
    data = response.json()
    # Тональность нейтральная
    assert data["sentiment"] == "neutral"
    # Предупреждение
    assert data["warning"] is not None

# Тест длинного текста (без превышения лимита)
def test_long_text():
    # Генерация длинного текста (500 повторов)
    long_review = "Very good movie. " * 500
    # POST-запрос
    response = client.post("/predict", json={"review": long_review})
    # Статус 200
    assert response.status_code == 200
    # Наличие поля sentiment
    assert "sentiment" in response.json()

# Тест отсутствия обязательного поля в запросе
def test_missing_field():
    # POST-запрос с пустым JSON
    response = client.post("/predict", json={})
    # Ошибка валидации Pydantic (422)
    assert response.status_code == 422

# Тест диапазона уверенности (от 0 до 1)
def test_confidence_range():
    # POST-запрос с нейтральным текстом
    response = client.post("/predict", json={"review": "A nice little indie film with heart."})
    # Статус
    assert response.status_code == 200
    # Уверенность
    conf = response.json()["confidence"]
    # Проверка границ
    assert 0.0 <= conf <= 1.0

# Тест предупреждения о слишком длинном тексте (>10000 символов)
def test_warning_on_too_long():
    # Генерация текста длиной 10001 символ
    long_review = "a" * 10001
    # POST-запрос
    response = client.post("/predict", json={"review": long_review})
    # Статус
    assert response.status_code == 200
    # Данные
    data = response.json()
    # Предупреждение
    assert data["warning"] is not None
    # Проверка наличия слова "длин" или "10" в предупреждении
    assert "длин" in data["warning"].lower() or "10" in data["warning"]