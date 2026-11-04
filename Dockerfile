# Imagen de la API REST de Detector IA.
# Construir:  docker build -t detector-ia .
# Ejecutar:   docker run -p 8000:8000 detector-ia

FROM python:3.11-slim

WORKDIR /app

# Primero los metadatos para aprovechar la caché de capas.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY api ./api

RUN pip install --no-cache-dir ".[api]"

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
