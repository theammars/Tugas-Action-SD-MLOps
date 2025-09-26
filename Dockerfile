FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "uvicorn[standard]" fastapi

COPY app ./app
COPY model ./model

ENV MODEL_PATH=/app/model/artifacts/latest/stroke_pipeline.joblib

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
