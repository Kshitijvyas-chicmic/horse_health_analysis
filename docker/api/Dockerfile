# --- Horse Health API: Dockerfile for Production Inference ---
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/mmpose

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything
COPY . .

# Expose port 8000
EXPOSE 8000

# Run API
CMD ["gunicorn", "-w", "4", "-k", uvicorn.workers.UvicornWorker", "apis.main:app", "--bind", "0.0.0.0:8000", "--timeout", "120"]
