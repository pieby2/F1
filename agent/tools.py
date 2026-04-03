"""
agent/tools.py – Tool definitions for the F1 prediction agent.

Each tool wraps a capability of the system (predict, explain, preview,
retrain) behind a simple callable interface compatible with LangChain-style
tool usage (name, description, args schema, __call__).

TODO: When OPENAI_API_KEY is set, the agent in agent.py uses these tools to
answer natural-language questions.
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

import httpx
from loguru import logger

# Default API base URL (overridable via env)
API_BASE = os.getenv("F1_API_BASE", "http://localhost:8000")


# ── Helper ────────────────────────────────────────────────────────────────────

def _api_post(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    with httpx.Client(timeout=30) as client:
        resp = client.post(f"{API_BASE}{endpoint}", json=payload)
        resp.raise_for_status()
        return resp.json()  # type: ignore[return-value]


def _api_get(endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    with httpx.Client(timeout=30) as client:
        resp = client.get(f"{API_BASE}{endpoint}", params=params)
        resp.raise_for_status()
        return resp.json()  # type: ignore[return-value]


# ── Tool: predict_race ────────────────────────────────────────────────────────

PREDICT_RACE_DESCRIPTION = """
Predict finishing positions for all drivers in an F1 race.
Input: JSON with keys circuit_id (str), season (int), round (int),
drivers (list of driver dicts with driver_id, constructor_id,
qualifying_position fields).
Output: sorted list of driver predictions with predicted_position,
podium_prob, top10_prob.
"""


def predict_race(
    circuit_id: str,
    season: int,
    round_num: int,
    drivers: list[dict[str, Any]],
    is_wet: int = 0,
) -> dict[str, Any]:
    """Call the /predict endpoint and return race predictions."""
    payload = {
        "circuit_id": circuit_id,
        "season": season,
        "round": round_num,
        "drivers": drivers,
        "is_wet": is_wet,
    }
    logger.info(f"[tool:predict_race] circuit={circuit_id} season={season} round={round_num}")
    return _api_post("/predict", payload)


# ── Tool: explain_prediction ──────────────────────────────────────────────────

EXPLAIN_DESCRIPTION = """
Explain why the model gave a particular finishing position prediction for one
driver. Returns feature importances and the driver's feature values.
Input: driver_id (str), constructor_id (str), grid (int),
qualifying_position (int), circuit_id (str), season (int), round (int).
"""


def explain_prediction(
    driver_id: str,
    constructor_id: str,
    grid: int,
    qualifying_position: int,
    circuit_id: str,
    season: int,
    round_num: int,
) -> dict[str, Any]:
    """Call the /explain endpoint for a single driver."""
    payload = {
        "driver_id": driver_id,
        "constructor_id": constructor_id,
        "grid": grid,
        "qualifying_position": qualifying_position,
        "circuit_id": circuit_id,
        "season": season,
        "round": round_num,
    }
    logger.info(f"[tool:explain_prediction] driver={driver_id} circuit={circuit_id}")
    return _api_post("/explain", payload)


# ── Tool: generate_preview ────────────────────────────────────────────────────

PREVIEW_DESCRIPTION = """
Generate a natural-language race preview for an upcoming Grand Prix, including
predicted finishing order, favourite drivers, and notable storylines.
Input: circuit_id (str), season (int), round (int),
drivers (list of driver dicts same as predict_race).
Output: dict with 'preview_text' and 'predictions' list.
"""


def generate_preview(
    circuit_id: str,
    season: int,
    round_num: int,
    drivers: list[dict[str, Any]],
    is_wet: int = 0,
) -> dict[str, Any]:
    """Call the /preview endpoint to get a race preview."""
    payload = {
        "circuit_id": circuit_id,
        "season": season,
        "round": round_num,
        "drivers": drivers,
        "is_wet": is_wet,
    }
    logger.info(f"[tool:generate_preview] circuit={circuit_id}")
    return _api_post("/preview", payload)


# ── Tool: retrain_pipeline ────────────────────────────────────────────────────

RETRAIN_DESCRIPTION = """
Trigger a full retrain of the F1 race prediction model using the latest data.
This runs the Prefect pipeline: ingest → features → train → register.
Returns a status message with the MLflow run ID on success.
WARNING: This is a long-running operation (minutes).
"""


def retrain_pipeline(seasons: list[int] | None = None) -> dict[str, Any]:
    """
    Trigger the training pipeline.

    If seasons is provided, only those seasons are re-ingested. Otherwise the
    pipeline uses the seasons defined in configs/config.yaml.

    This is implemented as a subprocess call so it can be triggered from the
    agent without blocking the API worker indefinitely.

    TODO: Replace with a proper Prefect deployment trigger for production.
    """
    logger.info("[tool:retrain_pipeline] Triggering pipeline…")
    cmd = [sys.executable, "-m", "flows.pipeline"]
    if seasons:
        cmd += ["--seasons"] + [str(s) for s in seasons]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            return {"status": "success", "output": result.stdout[-2000:]}
        return {"status": "error", "output": result.stderr[-2000:]}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "output": "Pipeline exceeded 600s timeout."}
    except Exception as exc:
        return {"status": "error", "output": str(exc)}


# ── Tool: health_check ────────────────────────────────────────────────────────

def health_check() -> dict[str, Any]:
    """Return the /health status of the prediction API."""
    return _api_get("/health")
