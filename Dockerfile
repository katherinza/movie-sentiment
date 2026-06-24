# Установка базового образа Python 3.11 (легковесная версия)
FROM python:3.11-slim

# Установка рабочей директории внутри контейнера
WORKDIR /app

# Копирование файла зависимостей в контейнер
COPY requirements.txt .
# Установка Python-пакетов из requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода приложения в контейнер
COPY app/ ./app/
# Копирование сохранённой модели машинного обучения
COPY model.joblib .

# Открытие порта 8000 для внешнего доступа
EXPOSE 8000

# Команда запуска FastAPI-приложения через Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]