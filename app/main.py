from __future__ import annotations

import os
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Any

import fastf1
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.inference import InferenceService


class PredictRaceRequest(BaseModel):
    season: int = Field(..., ge=1950)
    round: int = Field(..., ge=1)


class IngestDataRequest(BaseModel):
    season: int = Field(..., ge=2018)
    start_round: int = Field(1, ge=1)
    end_round: int | None = None


class DriverPrediction(BaseModel):
    driver_code: str
    driver_name: str
    team: str
    grid_position: int | None
    predicted_finish: int
    p_win: float
    p_podium: float
    p_points: float
    p_dnf: float



class RacePredictionResponse(BaseModel):
    season: int
    round: int
    event_name: str
    event_date: str | None
    feature_source: str
    most_likely_winner: str
    predicted_podium: list[str]
    drivers: list[DriverPrediction]
@lru_cache(maxsize=1)
def get_service() -> InferenceService:
    data_root = os.getenv("DATA_ROOT", "data/fastf1_csv")
    models_root = os.getenv("MODELS_ROOT", "models")
    cache_dir = os.getenv("INFERENCE_CACHE_DIR", "data/fastf1_csv/_api_cache")
    return InferenceService(data_root=data_root, models_root=models_root, cache_dir=cache_dir)



app = FastAPI(
    title="F1 Race Prediction API",
    description="Post-qualifying race prediction API with walk-forward retraining.",
    version="0.2.0",
)


@app.on_event("startup")
def startup_warm_cache() -> None:
    enabled = os.getenv("PREWARM_INFERENCE", "1").strip().lower() in {"1", "true", "yes", "on"}
    if enabled:
        get_service()


allow_origins = [item.strip() for item in os.getenv("ALLOW_ORIGINS", "*").split(",") if item.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/seasons")
def seasons() -> dict[str, list[int]]:
    service = get_service()
    values = service.get_available_seasons()

    # Also include current + next year from FastF1 schedule
    from datetime import datetime, timezone
    current_year = datetime.now(timezone.utc).year
    for extra_year in [current_year, current_year + 1]:
        if extra_year not in values:
            try:
                schedule = fastf1.get_event_schedule(extra_year, include_testing=False)
                if len(schedule) > 0:
                    values.append(extra_year)
            except Exception:
                pass

    values = sorted(set(values))
    return {"seasons": values}


@app.get("/events/next")
def next_event() -> dict[str, Any]:
    service = get_service()
    item = service.get_next_event()
    if item is None:
        raise HTTPException(status_code=404, detail="Could not determine next event from schedule.")
    return item


@app.get("/events/{season}")
def events(season: int) -> dict[str, Any]:
    service = get_service()
    values = service.get_events(season)

    # If no events from data, try FastF1 schedule directly
    if not values:
        try:
            schedule = fastf1.get_event_schedule(season, include_testing=False)
            for _, row in schedule.iterrows():
                rn = int(row["RoundNumber"]) if row["RoundNumber"] > 0 else None
                if rn is None:
                    continue
                date_val = None
                try:
                    import pandas as pd
                    d = pd.to_datetime(row["EventDate"], errors="coerce")
                    date_val = d.date().isoformat() if pd.notna(d) else None
                except Exception:
                    pass
                values.append({
                    "round": rn,
                    "event_name": str(row["EventName"]),
                    "date": date_val,
                    "available_for_prediction": False,
                    "has_race_context": False,
                })
        except Exception:
            pass

    return {"season": int(season), "events": values}


@app.post("/predict_race", response_model=RacePredictionResponse)
def predict_race(payload: PredictRaceRequest) -> RacePredictionResponse:
    service = get_service()

    try:
        prediction = service.predict_race(season=payload.season, round_number=payload.round)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    drivers = prediction.get("drivers", [])
    if not drivers:
        raise HTTPException(status_code=404, detail="No driver rows available for this race.")

    winner = max(drivers, key=lambda item: item.get("p_win", 0.0))
    podium = [item["driver_code"] for item in sorted(drivers, key=lambda item: item["predicted_finish"])[:3]]

    return RacePredictionResponse(
        season=int(prediction["season"]),
        round=int(prediction["round"]),
        event_name=str(prediction["event_name"]),
        event_date=prediction.get("event_date"),
        feature_source=str(prediction.get("feature_source", "unknown")),
        most_likely_winner=str(winner["driver_code"]),
        predicted_podium=podium,
        drivers=drivers,
    )


@app.post("/upload_models")
async def upload_models(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload a ZIP file of retrained models to replace the existing ones."""
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are supported.")

    models_root = Path(os.getenv("MODELS_ROOT", "models"))
    models_root.mkdir(parents=True, exist_ok=True)

    # Save zip to a temp file
    temp_zip = models_root / "temp_upload.zip"
    try:
        with open(temp_zip, "wb") as f:
            while contents := await file.read(1024 * 1024):
                f.write(contents)

        # Extract it
        import zipfile
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            # Basic validation to ensure we're extracting joblibs
            for member in zip_ref.namelist():
                if not member.endswith(".joblib") and not member.endswith("/"):
                    continue
                zip_ref.extract(member, models_root)
        
        # Invalidate InferenceService cache to load new models!
        get_service.cache_clear()
        
        return {"status": "success", "message": "Models uploaded and cache cleared."}
        
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process upload: {str(exc)}") from exc
    finally:
        if temp_zip.exists():
            temp_zip.unlink()


@app.post("/ingest_data")
def ingest_data(payload: IngestDataRequest) -> dict[str, Any]:
    """Trigger FastF1 data ingestion for a season/round range."""
    from fastf1_csv_ingest import ingest_single_round, get_round_numbers

    data_root = Path(os.getenv("DATA_ROOT", "data/fastf1_csv"))
    cache_dir = data_root / "fastf1_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    try:
        rounds = get_round_numbers(payload.season)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot load schedule: {exc}") from exc

    rounds = [r for r in rounds if r >= payload.start_round]
    if payload.end_round is not None:
        rounds = [r for r in rounds if r <= payload.end_round]

    results = {}
    for rnd in rounds:
        try:
            result = ingest_single_round(payload.season, rnd, data_root, overwrite=False)
            results[f"round_{rnd}"] = result
        except Exception as exc:
            results[f"round_{rnd}"] = {"error": str(exc)}

    # Clear cache so new data is picked up
    get_service.cache_clear()

    return {
        "season": payload.season,
        "rounds_processed": len(rounds),
        "results": results,
    }
