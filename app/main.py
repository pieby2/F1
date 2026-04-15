from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
import tempfile
import zipfile
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query
from pydantic import BaseModel, Field

from src.history_service import Formula1HistoryService
from src.inference import InferenceService
from src.news_service import Formula1NewsService


class PredictRaceRequest(BaseModel):
    season: int = Field(..., ge=1950)
    round: int = Field(..., ge=1)


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


class NewsArticleSummary(BaseModel):
    title: str
    summary: str
    source: str
    url: str
    published_at: str | None


class NewsDigestResponse(BaseModel):
    query: str
    count: int
    generated_at: str
    overall_summary: str
    top_topics: list[str]
    articles: list[NewsArticleSummary]


class HistoryDriverSummary(BaseModel):
    driver_code: str
    driver_name: str
    team: str
    races: int
    wins: int
    losses: int
    podiums: int
    points: float
    dnfs: int
    avg_finish: float | None
    best_finish: int | None
    teammate_wins: int
    teammate_losses: int
    summary: str


class HistoryTeamSummary(BaseModel):
    team: str
    races: int
    wins: int
    losses: int
    podiums: int
    points: float
    dnfs: int
    avg_finish: float | None
    best_finish: int | None
    drivers: list[str]
    summary: str


class HistoryDigestResponse(BaseModel):
    requested_season: int
    resolved_season: int
    generated_at: str
    availability_note: str
    available_seasons: list[int]
    overall_summary: str
    highlights: list[str]
    drivers: list[HistoryDriverSummary]
    teams: list[HistoryTeamSummary]


@lru_cache(maxsize=1)
def get_service() -> InferenceService:
    data_root = os.getenv("DATA_ROOT", "data/fastf1_csv")
    models_root = os.getenv("MODELS_ROOT", "models")
    cache_dir = os.getenv("INFERENCE_CACHE_DIR", "data/fastf1_csv/_api_cache")
    return InferenceService(data_root=data_root, models_root=models_root, cache_dir=cache_dir)


@lru_cache(maxsize=1)
def get_news_service() -> Formula1NewsService:
    query = os.getenv("F1_NEWS_QUERY", '"Formula 1" OR F1 when:7d')
    feed_url = os.getenv("F1_NEWS_FEED_URL", "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en")
    timeout = float(os.getenv("F1_NEWS_TIMEOUT_SECONDS", "12"))
    cache_ttl = int(os.getenv("F1_NEWS_CACHE_TTL_SECONDS", "900"))
    return Formula1NewsService(
        query=query,
        feed_url_template=feed_url,
        timeout_seconds=timeout,
        cache_ttl_seconds=cache_ttl,
    )


@lru_cache(maxsize=1)
def get_history_service() -> Formula1HistoryService:
    data_root = os.getenv("DATA_ROOT", "data/fastf1_csv")
    cache_dir = os.getenv("HISTORY_CACHE_DIR", "data/fastf1_csv/_api_cache")
    return Formula1HistoryService(data_root=data_root, cache_dir=cache_dir)


app = FastAPI(
    title="F1 Race Prediction API",
    description="Post-qualifying race prediction API powered by trained FastF1 models.",
    version="0.1.0",
)


@app.on_event("startup")
def startup_warm_cache() -> None:
    enabled = os.getenv("PREWARM_INFERENCE", "1").strip().lower() in {"1", "true", "yes", "on"}
    if enabled:
        get_service()


@app.post("/upload_models")
async def upload_models(file: UploadFile = File(...)) -> dict[str, str]:
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Please upload a .zip archive containing model files.")

    models_root = Path(os.getenv("MODELS_ROOT", "models"))
    models_root.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded archive is empty.")

    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = Path(temp_dir) / "models_upload.zip"
        archive_path.write_bytes(content)

        try:
            with zipfile.ZipFile(archive_path) as archive:
                members = [name for name in archive.namelist() if name and not name.endswith("/")]
                if not members:
                    raise HTTPException(status_code=400, detail="Uploaded archive does not contain any files.")

                if not any(name.lower().endswith(".joblib") for name in members):
                    raise HTTPException(status_code=400, detail="Uploaded archive must contain .joblib model files.")

                root = models_root.resolve()
                for member in members:
                    member_path = (models_root / member).resolve()
                    if member_path != root and root not in member_path.parents:
                        raise HTTPException(status_code=400, detail=f"Archive contains an unsafe path: {member}")

                for existing in models_root.glob("*.joblib"):
                    existing.unlink(missing_ok=True)

                archive.extractall(models_root)
        except zipfile.BadZipFile as exc:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid zip archive.") from exc

    return {"message": "Models uploaded successfully."}

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
    return {"season": int(season), "events": values}


@app.get("/news/summary", response_model=NewsDigestResponse)
def news_summary(count: int = Query(default=6, ge=5, le=7)) -> NewsDigestResponse:
    service = get_news_service()

    try:
        payload = service.build_digest(count=count)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return NewsDigestResponse(**payload)


@app.get("/history/summary", response_model=HistoryDigestResponse)
def history_summary(season: int | None = Query(default=None, ge=1950)) -> HistoryDigestResponse:
    service = get_history_service()

    try:
        payload = service.build_history_summary(season=season)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return HistoryDigestResponse(**payload)


@app.post("/predict_race", response_model=RacePredictionResponse)
def predict_race(payload: PredictRaceRequest) -> RacePredictionResponse:
    service = get_service()

    try:
        prediction = service.predict_race(season=payload.season, round_number=payload.round)
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
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
