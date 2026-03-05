FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1 \
    PIP_ONLY_BINARY=:all:

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    ccache \
    curl \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    poppler-utils \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash vscode \
    && echo "vscode ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/vscode \
    && chmod 0440 /etc/sudoers.d/vscode

COPY .devcontainer/scripts/install_python_deps.sh /tmp/install_python_deps.sh
RUN bash /tmp/install_python_deps.sh && rm -f /tmp/install_python_deps.sh

WORKDIR /workspaces/OCR_pipeline
USER vscode
