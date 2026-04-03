"""api/routes/predict.py – /predict endpoint."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from src.inference import predict_race
from src.monitoring import log_prediction

router = APIRouter()


class DriverInput(BaseModel):
    driver_id: str
    driver_code: str = ""
    constructor_id: str
    grid: int = Field(default=10, ge=1, le=20)
    qualifying_position: int = Field(default=10, ge=1, le=20)
    form_avg_finish: float = 10.5
    form_avg_points: float = 5.0
    form_dnf_rate: float = 0.05
    constructor_standings_pos: int = 5
    constructor_pts_season: float = 100.0
    circuit_avg_finish: float = 10.5


class PredictRequest(BaseModel):
    circuit_id: str
    season: int = Field(default=2025, ge=1950, le=2100)
    round: int = Field(default=1, ge=1, le=30, alias="round")
    drivers: list[DriverInput]
    is_wet: int = Field(default=0, ge=0, le=1)
    total_rounds: int = 24

    class Config:
        populate_by_name = True


@router.post("/predict", tags=["prediction"])
def predict(req: PredictRequest) -> dict[str, Any]:
    """
    Predict finishing positions for all drivers in an upcoming race.

    Drivers should be listed with at least `driver_id`, `constructor_id`, and
    `qualifying_position`. All other fields default to mid-field values.
    """
    try:
        predictions = predict_race(
            drivers=[d.model_dump() for d in req.drivers],
            circuit_id=req.circuit_id,
            season=req.season,
            round_num=req.round,
            total_rounds=req.total_rounds,
            is_wet=req.is_wet,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Log predictions for monitoring
    for pred in predictions:
        log_prediction(
            {
                "circuit_id": req.circuit_id,
                "season": req.season,
                "round": req.round,
                **pred,
            }
        )

    return {
        "circuit_id": req.circuit_id,
        "season": req.season,
        "round": req.round,
        "predictions": predictions,
    }
