FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY response_service response_service
COPY artifacts/model.json artifacts/model.json
RUN useradd --create-home service
USER service
EXPOSE 8000
CMD ["uvicorn", "response_service.api:app", "--host", "0.0.0.0", "--port", "8000"]
