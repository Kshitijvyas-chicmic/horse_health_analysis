# --- Horse Health Trainer: Dockerfile for GPU-Accelerated Training ---
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/mmpose

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

# Copy code
COPY . .

# Default command
ENTRYPOINT ["bash"]
