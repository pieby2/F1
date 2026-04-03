# F1 Race Prediction System

A full-stack F1 race prediction prototype with an MLOps pipeline and an agentic AI layer.

## Features

- **ETL pipeline** – fetches race results, qualifying, and standings from the Jolpica (Ergast-compatible) API and stores them as Parquet files.
- **Feature engineering** – builds a driver–race dataset with rolling form, constructor strength, and circuit-history features using DuckDB.
- **Training** – LightGBM regressor trained on finishing positions, logged and registered in MLflow.
- **Prediction API** – FastAPI with `/predict`, `/explain`, `/preview`, and `/health` endpoints.
- **Agentic AI** – stub agent (keyword routing) upgradeable to GPT via `OPENAI_API_KEY`.
- **Orchestration** – Prefect flow: ingest → features → train → register.
- **Monitoring** – Evidently data-drift and regression reports.
- **UI** – minimal Jinja2 HTML page to request race previews and see predictions.

---

## Quick Start (Docker Compose)

```bash
# 1. Clone and enter the repo
git clone https://github.com/pieby2/F1.git
cd F1

# 2. Create your .env file
cp .env.example .env
# Edit .env to add OPENAI_API_KEY (optional, for LLM-backed agent)

# 3. Start MLflow tracking server and prediction API
docker compose up --build

# 4. Open the UI
open http://localhost:8000

# 5. Open MLflow UI
open http://localhost:5000
```

The first time you start, no model is trained yet. Run the pipeline first (see below).

---

## Run the Pipeline (First Time or Retrain)

```bash
# Inside the container or locally with venv activated:
docker compose run --rm prefect python -m flows.pipeline

# Or locally:
python -m flows.pipeline --seasons 2022 2023 2024
```

This will:
1. Ingest race data from the Jolpica API.
2. Build the feature dataset.
3. Train a LightGBM model.
4. Register the model as `Production` in MLflow.

---

## Local Development (without Docker)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and edit env
cp .env.example .env

# Start MLflow (local file tracking – no server needed)
# Leave MLFLOW_TRACKING_URI blank or set to file:./mlruns in .env

# Run pipeline
python -m flows.pipeline --seasons 2022 2023 2024

# Start API
uvicorn api.main:app --reload --port 8000
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/`  | Minimal HTML UI |
| `GET`  | `/health` | System health check |
| `POST` | `/predict` | Predict all driver finishing positions |
| `POST` | `/explain` | Explain a single driver's prediction |
| `POST` | `/preview` | Race preview with narrative text |
| `POST` | `/agent`   | Natural-language agent query |

Interactive docs at `http://localhost:8000/docs`.

---

## Project Structure

```
F1/
├── api/
│   ├── main.py              # FastAPI app
│   ├── routes/              # Endpoint handlers
│   └── templates/           # Jinja2 HTML UI
├── agent/
│   ├── agent.py             # Stub + LLM agent
│   └── tools.py             # Tool callables
├── flows/
│   └── pipeline.py          # Prefect orchestration flow
├── src/
│   ├── ingest.py            # Data ingestion (Jolpica API)
│   ├── features.py          # Feature engineering
│   ├── train.py             # LightGBM training + MLflow
│   ├── inference.py         # Prediction + explanation
│   ├── monitoring.py        # Evidently drift reports
│   └── utils.py             # Config, logging, helpers
├── data/
│   ├── raw/                 # Raw Parquet from API
│   ├── processed/           # Merged/cleaned datasets
│   └── snapshots/           # Versioned feature snapshots
├── configs/
│   └── config.yaml          # All configuration
├── models/                  # MLflow model artifacts
├── mlruns/                  # MLflow tracking store
├── reports/
│   └── evidently/           # Drift + regression reports
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Configuration

All settings live in `configs/config.yaml`. Key options:

| Key | Default | Description |
|-----|---------|-------------|
| `mlflow.tracking_uri` | `http://localhost:5000` | MLflow server URL |
| `data.train_seasons` | `[2018..2023]` | Seasons used for training |
| `data.val_season` | `2024` | Season held out for validation |
| `features.form_window` | `5` | Rolling race window for driver form |
| `model.params.*` | LightGBM defaults | Hyperparameters |
| `agent.use_llm` | `false` | Set `true` + `OPENAI_API_KEY` for GPT agent |

---

## TODOs / Extending the Prototype

- [ ] **Real weather data** – integrate FastF1 or a weather API and set `is_wet` feature properly.
- [ ] **FastF1 telemetry** – add lap-time and sector-pace features for richer predictions.
- [ ] **Calibrated probabilities** – fit isotonic regression on validation predictions.
- [ ] **Hyperparameter tuning** – add Optuna sweep in `src/train.py`.
- [ ] **Full agentic loop** – define OpenAI function schemas in `agent/agent.py` for a true multi-step agent.
- [ ] **CI/CD** – add GitHub Actions workflow for automated retraining after each race weekend.
- [ ] **Cloud deployment** – replace Docker Compose with Fly.io / Render free tier.
- [ ] **Authentication** – add API key middleware for production.

---

## License

MIT
