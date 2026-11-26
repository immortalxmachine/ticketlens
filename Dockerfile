# CPU image for serving the API and the Gradio demo.
# Trained models are NOT baked in: mount ./models (see docker-compose.yml).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf_cache

WORKDIR /app

# CPU-only torch first (much smaller than the CUDA wheels), then the rest.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY ticketlens/ ticketlens/
COPY pyproject.toml .

EXPOSE 8020 7870
CMD ["uvicorn", "ticketlens.api:app", "--host", "0.0.0.0", "--port", "8020"]
