"""
src/train.py – Train a LightGBM regression model on the driver–race feature
dataset and log everything to MLflow.

Usage:
    python -m src.train
"""
from __future__ import annotations

from typing import Any

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.features import load_feature_snapshot
from src.utils import configure_logging, load_config, get_mlflow_tracking_uri, ensure_dirs, project_root

from loguru import logger

configure_logging()

# ── Feature columns used for training ────────────────────────────────────────

FEATURE_COLS: list[str] = [
    "grid",
    "qualifying_position",
    "form_avg_finish",
    "form_avg_points",
    "form_dnf_rate",
    "form_races_counted",
    "constructor_standings_pos",
    "constructor_pts_season",
    "circuit_avg_finish",
    "round_fraction",
    "is_wet",
    "circuit_id_enc",
]
TARGET_COL = "finish_position"


def _time_split(
    df: pd.DataFrame,
    val_season: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train (all seasons before val_season) and val."""
    train = df[df["season"] < val_season].copy()
    val = df[df["season"] == val_season].copy()
    logger.info(f"Train: {len(train)} rows | Val (season={val_season}): {len(val)} rows")
    return train, val


def _top_k_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 2,
) -> float:
    """Fraction of predictions within ±k positions of actual."""
    return float(np.mean(np.abs(y_true - y_pred) <= k))


def train(feature_df: pd.DataFrame | None = None) -> str:
    """
    Train the model and log to MLflow.

    Returns the MLflow run_id of the logged run.
    """
    cfg = load_config()
    model_cfg: dict[str, Any] = cfg["model"]["params"]
    val_season: int = cfg["data"]["val_season"]
    experiment_name: str = cfg["mlflow"]["experiment_name"]
    model_name: str = cfg["mlflow"]["model_name"]

    # ── Load features ──
    if feature_df is None:
        logger.info("Loading feature snapshot…")
        feature_df = load_feature_snapshot()

    train_df, val_df = _time_split(feature_df, val_season)

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COL].values
    X_val = val_df[FEATURE_COLS].values
    y_val = val_df[TARGET_COL].values

    # ── MLflow setup ──
    tracking_uri = get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run: {run_id}")

        # Log parameters
        mlflow.log_params(model_cfg)
        mlflow.log_param("val_season", val_season)
        mlflow.log_param("feature_cols", FEATURE_COLS)
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("val_rows", len(val_df))

        # ── Train ──
        model = lgb.LGBMRegressor(**model_cfg)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(50)],
        )

        # ── Evaluate ──
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        top1_acc = _top_k_accuracy(y_val, y_pred, k=1)
        top2_acc = _top_k_accuracy(y_val, y_pred, k=2)

        logger.info(f"Val MAE: {mae:.3f} | Top-1 Acc: {top1_acc:.3f} | Top-2 Acc: {top2_acc:.3f}")
        mlflow.log_metric("val_mae", mae)
        mlflow.log_metric("val_top1_acc", top1_acc)
        mlflow.log_metric("val_top2_acc", top2_acc)

        # ── Log feature importance ──
        fi = pd.DataFrame(
            {"feature": FEATURE_COLS, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)
        fi_path = project_root() / "reports" / "feature_importance.csv"
        ensure_dirs(fi_path.parent)
        fi.to_csv(fi_path, index=False)
        mlflow.log_artifact(str(fi_path), "feature_importance")

        # ── Log model ──
        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_name,
        )
        logger.info(f"Model registered as '{model_name}'")

    return run_id


if __name__ == "__main__":
    run_id = train()
    logger.info(f"Training complete. Run ID: {run_id}")
