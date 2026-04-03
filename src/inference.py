"""
src/inference.py – Load the registered MLflow model and run predictions.

Provides two main functions used by the FastAPI routes and agent tools:
  - predict_race(): predict finishing positions for a list of drivers.
  - explain_prediction(): return feature importances for a given driver.
"""
from __future__ import annotations

from typing import Any

import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from loguru import logger

from src.features import build_inference_row
from src.train import FEATURE_COLS
from src.utils import configure_logging, get_mlflow_tracking_uri, load_config

configure_logging()

_model: Any | None = None  # module-level model cache


def _load_model() -> Any:
    """Load the Production model from MLflow (cached after first load)."""
    global _model
    if _model is not None:
        return _model

    cfg = load_config()
    model_name = cfg["mlflow"]["model_name"]
    tracking_uri = get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)

    try:
        model_uri = f"models:/{model_name}/Production"
        _model = mlflow.lightgbm.load_model(model_uri)
        logger.info(f"Loaded Production model: {model_uri}")
    except Exception as exc:
        logger.warning(f"Could not load Production model ({exc}). Trying latest version…")
        try:
            model_uri = f"models:/{model_name}/latest"
            _model = mlflow.lightgbm.load_model(model_uri)
            logger.info(f"Loaded latest model: {model_uri}")
        except Exception as exc2:
            logger.error(f"Model load failed: {exc2}")
            raise RuntimeError(
                "No trained model found. Run `python -m src.train` first."
            ) from exc2

    return _model


def predict_race(
    drivers: list[dict[str, Any]],
    circuit_id: str,
    season: int,
    round_num: int,
    total_rounds: int = 24,
    is_wet: int = 0,
) -> list[dict[str, Any]]:
    """
    Predict finishing positions for all drivers in an upcoming race.

    Parameters
    ----------
    drivers:
        List of dicts with keys: driver_id, constructor_id, grid,
        qualifying_position, [optional form/circuit fields].
    circuit_id:
        Ergast circuit ID (e.g. "monza", "monaco").
    season, round_num:
        Race identifiers.
    total_rounds:
        Total rounds in the season (for round_fraction feature).
    is_wet:
        1 = wet race forecast, 0 = dry.

    Returns
    -------
    List of dicts sorted by predicted_position, each containing driver_id,
    predicted_position, podium_prob, top10_prob.
    """
    model = _load_model()

    rows: list[pd.DataFrame] = []
    for d in drivers:
        row = build_inference_row(
            driver_id=d["driver_id"],
            constructor_id=d["constructor_id"],
            grid=d.get("grid", d.get("qualifying_position", 10)),
            qualifying_position=d.get("qualifying_position", 10),
            circuit_id=circuit_id,
            season=season,
            round_num=round_num,
            total_rounds=total_rounds,
            form_avg_finish=d.get("form_avg_finish", 10.5),
            form_avg_points=d.get("form_avg_points", 5.0),
            form_dnf_rate=d.get("form_dnf_rate", 0.05),
            constructor_standings_pos=d.get("constructor_standings_pos", 5),
            constructor_pts_season=d.get("constructor_pts_season", 100.0),
            circuit_avg_finish=d.get("circuit_avg_finish", 10.5),
            is_wet=is_wet,
        )
        rows.append(row)

    X = pd.concat(rows, ignore_index=True)[FEATURE_COLS].values
    raw_preds = model.predict(X)

    results = []
    for i, d in enumerate(drivers):
        pred_pos = float(raw_preds[i])
        # Derive simple probabilities from predicted position
        podium_prob = float(np.clip(1 - (pred_pos - 1) / 10, 0, 1))
        top10_prob = float(np.clip(1 - (pred_pos - 1) / 15, 0, 1))
        results.append(
            {
                "driver_id": d["driver_id"],
                "driver_code": d.get("driver_code", d["driver_id"][:3].upper()),
                "constructor_id": d["constructor_id"],
                "predicted_position_raw": pred_pos,
                "podium_prob": round(podium_prob, 3),
                "top10_prob": round(top10_prob, 3),
            }
        )

    # Sort by raw predicted position to generate final classification
    results.sort(key=lambda x: x["predicted_position_raw"])
    for rank, r in enumerate(results, start=1):
        r["predicted_position"] = rank

    return results


def explain_prediction(
    driver_id: str,
    constructor_id: str,
    grid: int,
    qualifying_position: int,
    circuit_id: str,
    season: int,
    round_num: int,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Return feature importances and values for a single driver–race prediction.
    """
    model = _load_model()

    row = build_inference_row(
        driver_id=driver_id,
        constructor_id=constructor_id,
        grid=grid,
        qualifying_position=qualifying_position,
        circuit_id=circuit_id,
        season=season,
        round_num=round_num,
        **kwargs,
    )

    X = row[FEATURE_COLS].values
    pred_pos = float(model.predict(X)[0])

    importances = dict(zip(FEATURE_COLS, model.feature_importances_))
    feature_values = dict(zip(FEATURE_COLS, X[0].tolist()))

    return {
        "driver_id": driver_id,
        "predicted_position_raw": round(pred_pos, 2),
        "predicted_position": max(1, round(pred_pos)),
        "feature_importances": {
            k: int(v) for k, v in sorted(
                importances.items(), key=lambda x: x[1], reverse=True
            )
        },
        "feature_values": {k: round(float(v), 4) for k, v in feature_values.items()},
    }
