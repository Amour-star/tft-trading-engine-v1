FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies (torch already installed in base image)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Fail fast if deployment artifact missed the Python package used by runtime imports
RUN test -f data/database.py && test -f data/__init__.py

# Create directories
RUN mkdir -p logs data/historical saved_models

# Expose ports
EXPOSE 8000 8501

# Default: run the trading engine
CMD ["python", "scripts/run_engine.py"]
