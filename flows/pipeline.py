"""
flows/pipeline.py – Prefect orchestration flow.

Runs the full MLOps pipeline:
  1. Ingest race data from Jolpica API
  2. Build feature dataset
  3. Train LightGBM model and log to MLflow
  4. Register the model as "Production" in MLflow

Usage:
    python -m flows.pipeline                  # run all seasons from config
    python -m flows.pipeline --seasons 2023   # re-ingest a specific season
"""
from __future__ import annotations

import argparse
from typing import Any

import mlflow
from loguru import logger
from prefect import flow, task
from prefect.logging import get_run_logger

from src.features import build_feature_dataset
from src.ingest import ingest_seasons
from src.train import train
from src.utils import configure_logging, get_mlflow_tracking_uri, load_config

configure_logging()


# ── Tasks ─────────────────────────────────────────────────────────────────────

@task(name="ingest", retries=2, retry_delay_seconds=30)
def ingest_task(seasons: list[int]) -> dict[str, Any]:
    log = get_run_logger()
    log.info(f"Ingesting seasons: {seasons}")
    combined = ingest_seasons(seasons)
    log.info("Ingestion complete.")
    return {k: len(v) for k, v in combined.items()}


@task(name="build_features")
def features_task() -> int:
    log = get_run_logger()
    log.info("Building feature dataset…")
    df = build_feature_dataset()
    log.info(f"Features built: {len(df)} rows")
    return len(df)


@task(name="train_model")
def train_task() -> str:
    log = get_run_logger()
    log.info("Training model…")
    run_id = train()
    log.info(f"Training complete. MLflow run: {run_id}")
    return run_id


@task(name="register_model")
def register_task(run_id: str) -> None:
    """Transition the latest model version to 'Production' in MLflow."""
    log = get_run_logger()
    cfg = load_config()
    model_name = cfg["mlflow"]["model_name"]
    tracking_uri = get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.MlflowClient()
    try:
        # Get the model version associated with this run
        versions = client.search_model_versions(f"run_id='{run_id}'")
        if not versions:
            log.warning("No model version found for this run. Skipping registration.")
            return

        latest_version = versions[0].version
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage="Production",
            archive_existing_versions=True,
        )
        log.info(
            f"Model '{model_name}' v{latest_version} transitioned to Production."
        )
    except Exception as exc:
        log.error(f"Model registration failed: {exc}")
        # Non-fatal: pipeline still succeeded
        raise


# ── Flow ──────────────────────────────────────────────────────────────────────

@flow(name="f1_mlops_pipeline", log_prints=True)
def f1_pipeline(seasons: list[int] | None = None) -> str:
    """
    Full F1 MLOps pipeline: ingest → features → train → register.

    Returns the MLflow run_id of the trained model.
    """
    cfg = load_config()
    if seasons is None:
        seasons = cfg["data"]["train_seasons"] + [cfg["data"]["val_season"]]

    ingestion_counts = ingest_task(seasons)
    feature_rows = features_task(wait_for=[ingestion_counts])
    run_id = train_task(wait_for=[feature_rows])
    register_task(run_id, wait_for=[run_id])
    return run_id


# ── CLI entry ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the F1 MLOps pipeline.")
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=None,
        help="Seasons to ingest (e.g. 2022 2023 2024). Defaults to config.",
    )
    args = parser.parse_args()
    run_id = f1_pipeline(seasons=args.seasons)
    logger.info(f"Pipeline finished. Run ID: {run_id}")
