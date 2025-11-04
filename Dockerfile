# Dockerfile for Hugging Face Spaces deployment
# Optimized for ML workloads with full features

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY gdrive_config.py .

# Create outputs directory
RUN mkdir -p outputs

# Hugging Face Spaces uses port 7860 by default
ENV PORT=7860
EXPOSE 7860

# Enable all features (HF Spaces has 16GB RAM)
ENV ENABLE_LLM=1
ENV ENABLE_CLIP=1
ENV ENABLE_GRADCAM=1
ENV FORCE_CPU=0

# Optional: Pre-download models at build time (faster startup)
# ENV DISEASE_CKPT=outputs/derm_best.pt
# ENV XR_CKPT=outputs/xray_best.pt
# RUN python -c "from gdrive_config import ensure_models_downloaded; ensure_models_downloaded()"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

# Start the application
CMD ["python", "app.py"]

