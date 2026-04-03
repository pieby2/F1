"""api/routes/preview.py – /preview endpoint."""
from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from api.routes.predict import DriverInput
from src.inference import predict_race
from src.utils import load_config

router = APIRouter()


class PreviewRequest(BaseModel):
    circuit_id: str
    race_name: str = ""
    season: int = 2025
    round: int = Field(default=1, alias="round")
    drivers: list[DriverInput]
    is_wet: int = 0
    total_rounds: int = 24

    class Config:
        populate_by_name = True


def _generate_preview_text(
    race_name: str,
    circuit_id: str,
    predictions: list[dict[str, Any]],
    is_wet: bool,
) -> str:
    """
    Generate a simple rule-based race preview text.

    TODO: Replace with LLM call (e.g., OpenAI) when OPENAI_API_KEY is set.
    """
    top3 = predictions[:3]
    podium_str = ", ".join(
        f"P{r['predicted_position']} {r['driver_id'].upper()}" for r in top3
    )
    favourite = top3[0]["driver_id"].upper() if top3 else "Unknown"
    fav_podium_pct = int(top3[0]["podium_prob"] * 100) if top3 else 0

    weather_note = "With wet conditions expected, expect the unexpected." if is_wet else ""

    cfg = load_config()
    use_llm = cfg.get("agent", {}).get("use_llm", False)

    if use_llm:
        try:
            import openai  # noqa: PLC0415

            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            prompt = (
                f"Write a short, engaging F1 race preview for the "
                f"{race_name or circuit_id} Grand Prix. "
                f"The model predicts the following podium: {podium_str}. "
                f"{'Conditions are wet.' if is_wet else 'Conditions are dry.'} "
                "Include 2–3 sentences about what to watch for."
            )
            resp = client.chat.completions.create(
                model=cfg["agent"].get("openai_model", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            logger.warning(f"LLM preview generation failed: {exc}. Using template.")

    return (
        f"Welcome to the {race_name or circuit_id.replace('_', ' ').title()} Grand Prix preview! "
        f"Our model predicts the following podium: {podium_str}. "
        f"{favourite} leads the field with a {fav_podium_pct}% podium probability. "
        f"{weather_note} "
        "Check the full prediction table below for all 20 drivers."
    ).strip()


@router.post("/preview", tags=["prediction"])
def preview(req: PreviewRequest) -> dict[str, Any]:
    """
    Generate a race preview: model predictions + narrative text.
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
        logger.exception("Preview prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    preview_text = _generate_preview_text(
        race_name=req.race_name,
        circuit_id=req.circuit_id,
        predictions=predictions,
        is_wet=bool(req.is_wet),
    )

    return {
        "circuit_id": req.circuit_id,
        "race_name": req.race_name,
        "season": req.season,
        "round": req.round,
        "preview_text": preview_text,
        "predictions": predictions,
    }
