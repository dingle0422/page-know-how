FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml /app/
RUN pip install --upgrade pip && pip install .

COPY app /app/app

ENV STORE_DIR=/app/data \
    HOST=0.0.0.0 \
    PORT=8088

EXPOSE 8088
VOLUME ["/app/data"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8088"]
