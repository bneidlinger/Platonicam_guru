# Surveillance Design Assistant API
#
# Expects an Ollama instance reachable at OLLAMA_HOST (see docker-compose.yml
# for the sidecar wiring). The vector store and PDF corpus live on volumes so
# the image stays stateless.
FROM python:3.12-slim

WORKDIR /app

# Layer-cache dependencies separately from source
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV OLLAMA_HOST=http://ollama:11434 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

# Sessions are in-process: keep a single worker (scale with sticky routing)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
