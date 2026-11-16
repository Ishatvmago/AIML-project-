# Serving image: only the API, the trained model and the built UI.
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so Docker can cache this layer between code changes
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY app.py features.py ./
COPY model/ model/
COPY static/ static/

# The LLM is optional: inside Docker the app falls back to template explanations unless
# OLLAMA_URL points to a reachable Ollama server (e.g. http://host.docker.internal:11434).
ENV PORT=8000 \
    PYTHONUNBUFFERED=1 \
    PREDICTION_LOG=/app/logs/predictions.jsonl

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:' + __import__('os').environ.get('PORT', '8000') + '/health')"

# Render (and most hosts) pass the port in $PORT
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
