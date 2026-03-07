FROM python:3.9-slim

# Install system dependencies for OCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install AWS CLI v2
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf aws awscliv2.zip

# Copy existing pipeline code (relative to project root)
COPY src/ /app/src/
COPY production/ /app/production/
COPY requirements.txt /app/

# Copy cloud infrastructure
COPY dev-infra/cloud/ /app/cloud/

# Install Python dependencies
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

# Install additional cloud dependencies
RUN pip install --no-cache-dir boto3 requests

# Copy cloud configuration
COPY config.json /app/config.json

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import boto3; print('AWS healthy')" || exit 1

# Default entrypoint - can be overridden for different job types
ENTRYPOINT ["python", "-m", "cloud.worker.job_processor"]
