# LabelCheck — образ приложения (Streamlit UI + CLI). Секреты в образ НЕ
# попадают: .env отдаётся контейнеру через docker-compose env_file.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Зависимости отдельным слоем — пересборка при правке кода не тянет pip.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Код, конфиги, корпус (data/chunks.jsonl, демо-журнал). Что не нужно
# в образе (.env, .git, личные макеты, кэши) — в .dockerignore.
COPY . .

EXPOSE 8501

# headless: без попытки открыть браузер; 0.0.0.0: доступ с хоста;
# без файлового наблюдателя: в контейнере код не меняется на лету.
CMD ["streamlit", "run", "labelcheck/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--server.fileWatcherType=none", \
     "--browser.gatherUsageStats=false"]
