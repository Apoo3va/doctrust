FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/

EXPOSE 8000

CMD ["sh", "-c", "python src/ingestion/run.py --path data && uvicorn src.api:app --host 0.0.0.0 --port 8000"]