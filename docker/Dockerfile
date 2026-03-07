FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    gcc git curl ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY requirements_minimal.txt .
RUN pip install --no-cache-dir -r requirements_minimal.txt
COPY . .
# Smoke default
CMD ["python","tools/run_page_text.py","--help"]
