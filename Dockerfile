FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed by some ML libs (e.g. xgboost, lightgbm use libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching - only reinstalls if requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the trained model artifacts explicitly
COPY models/ ./models/

# Copy application source code
COPY src/ ./src/

# Ensure Python output is not buffered (logs show up in real time)
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.serving.main:app", "--host", "0.0.0.0", "--port", "8000"]