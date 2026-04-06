# ========================
# Stage 1: Build frontend
# ========================
FROM node:20-alpine AS frontend-build

WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --production=false

COPY web/ ./

# Build the Vite app with API pointing to /api (proxied by nginx)
ENV VITE_API_BASE_URL=/api
RUN npm run build

# ========================
# Stage 2: Backend runtime
# ========================
FROM python:3.11-slim AS backend

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DATA_ROOT=/app/data/fastf1_csv
ENV MODELS_ROOT=/app/models
ENV PREWARM_INFERENCE=1

WORKDIR /app

# Install system dependencies for LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (cached layer)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app ./app
COPY src ./src
COPY train_grid_position_model.py ./
COPY train_grid_ranking_model.py ./
COPY train_race_points_model.py ./
COPY train_dnf_model.py ./
COPY train_race_outcome_multiclass_model.py ./
COPY fastf1_csv_ingest.py ./
COPY _prebuild_cache.py ./

# Copy pre-trained models and data
COPY models ./models
COPY data ./data

# Copy built frontend for nginx stage
COPY --from=frontend-build /web/dist /app/frontend-dist

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
