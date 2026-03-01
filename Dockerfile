# syntax=docker/dockerfile:1.7

ARG USE_CUDA=false

# ----------------------------
# Base images (CPU vs CUDA)
# ----------------------------

FROM python:3.11-slim AS base-false
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS base-true
ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv && \
    python3.11 -m ensurepip --upgrade && \
    python3.11 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3 && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get purge -y --auto-remove software-properties-common || true


# ----------------------------
# Builder (installs deps into venv)
# ----------------------------

FROM base-${USE_CUDA} AS builder
ARG USE_CUDA=false
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
      gcc g++ build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    grep -vE '^torch(==|>=|<=|~=|$)' /app/requirements.txt > /tmp/requirements.no-torch.txt && \
    if [ "$USE_CUDA" = "true" ]; then \
      python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1; \
    else \
      python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1; \
    fi && \
    python -m pip install --no-cache-dir -r /tmp/requirements.no-torch.txt

COPY . /app
RUN test -f data/database.py && test -f data/__init__.py


# ----------------------------
# Runtime (minimal, non-root)
# ----------------------------

FROM base-${USE_CUDA} AS runtime
ARG USE_CUDA=false
WORKDIR /app

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
      bash curl && \
    rm -rf /var/lib/apt/lists/*

RUN adduser --disabled-password --gecos "" --home /home/appuser --uid 10001 appuser

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app
COPY entrypoint.sh /app/entrypoint.sh

RUN mkdir -p /app/logs /app/data /app/state /app/saved_models && \
    chown -R appuser:appuser /app/logs /app/data /app/state /app/saved_models && \
    chmod +x /app/entrypoint.sh

USER appuser

EXPOSE 8000

STOPSIGNAL SIGTERM

ENTRYPOINT ["/app/entrypoint.sh"]
