"""api/routes/explain.py – /explain endpoint."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from src.inference import explain_prediction

router = APIRouter()


class ExplainRequest(BaseModel):
    driver_id: str
    constructor_id: str
    grid: int = Field(default=10, ge=1, le=20)
    qualifying_position: int = Field(default=10, ge=1, le=20)
    circuit_id: str
    season: int = 2025
    round: int = Field(default=1, alias="round")
    form_avg_finish: float = 10.5
    form_avg_points: float = 5.0
    form_dnf_rate: float = 0.05
    constructor_standings_pos: int = 5
    constructor_pts_season: float = 100.0
    circuit_avg_finish: float = 10.5
    is_wet: int = 0

    class Config:
        populate_by_name = True


@router.post("/explain", tags=["prediction"])
def explain(req: ExplainRequest) -> dict[str, Any]:
    """
    Explain the prediction for a single driver–race by returning feature
    importances and their input values.
    """
    try:
        result = explain_prediction(
            driver_id=req.driver_id,
            constructor_id=req.constructor_id,
            grid=req.grid,
            qualifying_position=req.qualifying_position,
            circuit_id=req.circuit_id,
            season=req.season,
            round_num=req.round,
            form_avg_finish=req.form_avg_finish,
            form_avg_points=req.form_avg_points,
            form_dnf_rate=req.form_dnf_rate,
            constructor_standings_pos=req.constructor_standings_pos,
            constructor_pts_season=req.constructor_pts_season,
            circuit_avg_finish=req.circuit_avg_finish,
            is_wet=req.is_wet,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Explanation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result
