# ─────────────────────────────────────────────
# STAGE 1: Build dependencies
# ─────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /app

# System deps needed for pandas, scipy, scikit-learn
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ─────────────────────────────────────────────
# STAGE 2: Final runtime image
# ─────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder — no build tools in final image
COPY --from=builder /usr/local /usr/local

# Copy project files
COPY . .

# MLflow / DagsHub — injected at runtime via docker run -e
# Never hardcode credentials in the image
ENV PYTHONUNBUFFERED=1

# FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]