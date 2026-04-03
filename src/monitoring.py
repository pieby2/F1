"""
src/monitoring.py – Data drift and prediction quality monitoring using Evidently.

Generates HTML reports comparing reference (training) data to current
prediction logs. Reports are saved to reports/evidently/.

TODO: Wire up a periodic Prefect task to run this and alert on drift.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from src.utils import configure_logging, ensure_dirs, project_root

configure_logging()

REPORT_DIR = project_root() / "reports" / "evidently"


def run_data_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: Path | None = None,
) -> Path:
    """
    Generate an Evidently data drift HTML report comparing *reference* to
    *current* DataFrames.

    Returns the path to the generated HTML report.
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except ImportError:
        logger.error("evidently is not installed. Add it to requirements.txt.")
        raise

    ensure_dirs(REPORT_DIR)
    if output_path is None:
        output_path = REPORT_DIR / "drift_report.html"

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html(str(output_path))
    logger.info(f"Drift report saved → {output_path}")
    return output_path


def run_regression_quality_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    target_col: str = "finish_position",
    prediction_col: str = "predicted_position_raw",
    output_path: Path | None = None,
) -> Path:
    """
    Generate an Evidently regression performance report.
    Both DataFrames must contain *target_col* and *prediction_col*.
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import RegressionPreset
        from evidently import ColumnMapping
    except ImportError:
        logger.error("evidently is not installed.")
        raise

    ensure_dirs(REPORT_DIR)
    if output_path is None:
        output_path = REPORT_DIR / "regression_report.html"

    column_mapping = ColumnMapping(
        target=target_col,
        prediction=prediction_col,
    )

    report = Report(metrics=[RegressionPreset()])
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )
    report.save_html(str(output_path))
    logger.info(f"Regression report saved → {output_path}")
    return output_path


def log_prediction(
    prediction_record: dict[str, Any],
    log_path: Path | None = None,
) -> None:
    """
    Append a single prediction record to the prediction log (Parquet).
    Used to build up the 'current' dataset for Evidently monitoring.
    """
    ensure_dirs(REPORT_DIR)
    if log_path is None:
        log_path = project_root() / "data" / "prediction_log.parquet"

    df_new = pd.DataFrame([prediction_record])
    if log_path.exists():
        df_existing = pd.read_parquet(log_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_parquet(log_path, index=False)


if __name__ == "__main__":
    # Example: generate a drift report using a tiny synthetic dataset
    import numpy as np

    rng = np.random.default_rng(42)
    ref = pd.DataFrame(
        {
            "grid": rng.integers(1, 20, 200),
            "qualifying_position": rng.integers(1, 20, 200),
            "form_avg_finish": rng.uniform(1, 20, 200),
        }
    )
    cur = pd.DataFrame(
        {
            "grid": rng.integers(1, 20, 50),
            "qualifying_position": rng.integers(1, 20, 50),
            "form_avg_finish": rng.uniform(5, 20, 50),  # slight drift
        }
    )
    run_data_drift_report(ref, cur)
