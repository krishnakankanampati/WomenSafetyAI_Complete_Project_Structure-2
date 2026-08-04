# Model weights (saved_models/) are NOT baked into this image - they're
# mounted as a volume at runtime (see docker-compose.yml / k8s manifests).
# Baking in a 1.1GB checkpoint would make every code-only change rebuild and
# repush that same gigabyte, which is the opposite of what CI should do.
FROM python:3.12-slim AS base

WORKDIR /app

# torch's CPU wheel comes from its own index - this host has no GPU, and the
# default PyPI wheel is the much larger CUDA build.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY ai_model/ ai_model/
COPY backend/ backend/
COPY database/ database/
COPY email_service/ email_service/
COPY social_media/ social_media/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
