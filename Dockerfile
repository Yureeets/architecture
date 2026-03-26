# Stage 1: builder — встановлення важких залежностей
FROM python:3.10 AS builder

WORKDIR /build

COPY requirements.txt .

# Встановлюємо залежності у /install, щоб потім скопіювати в slim-образ
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

# Stage 2: final — легкий образ для запуску пайплайну
FROM python:3.10-slim AS final

WORKDIR /app

# Копіюємо встановлені пакети з builder-стадії
COPY --from=builder /install /usr/local

# Копіюємо лише необхідний код
COPY src/ ./src/
COPY config/ ./config/
COPY dvc.yaml .

# Визначаємо шлях до даних через змінну середовища
ENV DATA_PATH=data/raw/hmnist_28_28_L.csv

# За замовчуванням запускаємо train_pipeline
CMD ["python", "src/train_pipeline.py"]
