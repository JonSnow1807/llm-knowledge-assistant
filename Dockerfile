# Fixed Dockerfile with proper permissions
FROM python:3.10-slim as builder

# Build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.10-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create user and set up directories
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /home/appuser/app

# Create cache directories with correct permissions
RUN mkdir -p /home/appuser/.cache/huggingface && \
    mkdir -p /home/appuser/.cache/torch && \
    chown -R appuser:appuser /home/appuser/.cache

# Copy application code
COPY --chown=appuser:appuser . .

# Set environment variables for cache
ENV HF_HOME=/home/appuser/.cache/huggingface
ENV TORCH_HOME=/home/appuser/.cache/torch
ENV TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Expose port
EXPOSE 5000

# Run gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "300", "--worker-class", "sync", "app:app"]
