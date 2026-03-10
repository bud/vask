FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libportaudio2 \
        libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir . && \
    pip install --no-cache-dir uvicorn[standard]

# Copy application code
COPY vask/ ./vask/

# Non-root user for security
RUN useradd --create-home --shell /bin/bash vask
USER vask

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8420/health'); r.raise_for_status()"

EXPOSE 8420

ENTRYPOINT ["python", "-m", "vask"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8420", "--log-json"]
