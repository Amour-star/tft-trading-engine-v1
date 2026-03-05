# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      gcc \
      g++ \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /app/requirements.txt


FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Europe/Berlin

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      bash \
      curl \
      libgomp1 \
      tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY --from=builder /opt/venv /opt/venv

COPY . /app

RUN adduser --disabled-password --gecos "" --home /home/appuser --uid 10001 appuser && \
    mkdir -p /app/logs /app/data /app/state /app/saved_models && \
    chown -R appuser:appuser /app/logs /app/data /app/state /app/saved_models

USER appuser

EXPOSE 8000 8501
STOPSIGNAL SIGTERM
CMD ["python", "scripts/run_engine.py"]
