"""api/routes/health.py – Health check endpoint."""
from __future__ import annotations

from typing import Any

import mlflow
from fastapi import APIRouter
from loguru import logger

from src.utils import get_mlflow_tracking_uri, load_config

router = APIRouter()


@router.get("/health", tags=["system"])
def health() -> dict[str, Any]:
    """Return service health status and basic connectivity checks."""
    cfg = load_config()
    tracking_uri = get_mlflow_tracking_uri()

    mlflow_ok = False
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.search_experiments()
        mlflow_ok = True
    except Exception as exc:
        logger.warning(f"MLflow connectivity check failed: {exc}")

    return {
        "status": "ok",
        "version": "0.1.0",
        "mlflow_uri": tracking_uri,
        "mlflow_reachable": mlflow_ok,
        "model_name": cfg["mlflow"]["model_name"],
    }
