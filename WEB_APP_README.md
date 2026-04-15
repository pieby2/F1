# F1 Race Prediction Website (v1)

This project now includes:

- A FastAPI backend for race prediction inference.
- A React + Vite frontend dashboard.

## Backend API

### Files

- `app/main.py`
- `src/inference/service.py`

### Endpoints

- `GET /health`
- `GET /seasons`
- `GET /events/{season}`
- `GET /events/next`
- `GET /news/summary?count=5..7`
- `GET /history/summary?season=YYYY`
- `POST /predict_race`

### Run locally

```bash
C:/Users/VICTUS/anaconda3/envs/swinfusion/python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8080
```

### Optional backend env vars

- `DATA_ROOT` (default: `data/fastf1_csv`)
- `MODELS_ROOT` (default: `models`)
- `INFERENCE_CACHE_DIR` (default: `data/fastf1_csv/_api_cache`)
- `INFERENCE_ENABLE_DATASET_CACHE` (default: `1`)
- `PREWARM_INFERENCE` (default: `1`) preloads models/features during API startup.
- `F1_NEWS_QUERY` (default: `"Formula 1" OR F1 when:7d`) controls the news search query.
- `F1_NEWS_FEED_URL` (default: Google News RSS search endpoint) controls the news feed source.
- `F1_NEWS_TIMEOUT_SECONDS` (default: `12`) controls the RSS fetch timeout.
- `F1_NEWS_CACHE_TTL_SECONDS` (default: `900`) keeps the news digest fresh while avoiding repeated fetches.
- `HISTORY_CACHE_DIR` (default: `data/fastf1_csv/_api_cache`) stores the local SQL cache used by the driver/team history endpoint.

### Request example

```bash
curl -X POST http://localhost:8080/predict_race \
  -H "Content-Type: application/json" \
  -d '{"season": 2025, "round": 8}'
```

## Frontend website

### Files

- `web/src/App.jsx`
- `web/src/api.js`
- `web/src/styles.css`

### Run locally

```bash
cd web
npm install
npm run dev
```

Open `http://localhost:5173`.

## Docker

### Backend container

```bash
docker build -t f1-race-api .
docker run -p 8080:8080 f1-race-api
```

### Frontend container

```bash
docker build -t f1-race-web ./web
docker run -p 5173:80 f1-race-web
```

## Notes

- Event dates are sourced from FastF1 schedule (`fastf1.get_event_schedule`) when available.
- If full context features are unavailable for a race, inference falls back to grid-only feature mode.
- The dashboard includes a Formula 1 news digest that summarizes 5 to 7 recent headlines and refreshes on demand.
- The dashboard also includes a driver/team history panel backed by a local SQL cache. If the selected season is newer than the cached race results, the API returns the latest ingested season and shows a note in the UI.
- The prediction API falls back to a grid-based heuristic when no trained models are mounted.
- The API now caches prebuilt feature tables in `data/fastf1_csv/_api_cache` to reduce first-request latency.
