"""
src/utils.py – Shared utilities: config loading, logging, DuckDB helpers.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

# ── Config ─────────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
_config: dict[str, Any] | None = None


def load_config(path: Path = _CONFIG_PATH) -> dict[str, Any]:
    """Load and cache the YAML config."""
    global _config
    if _config is None:
        with open(path) as fh:
            _config = yaml.safe_load(fh)
    return _config


# ── Logging ─────────────────────────────────────────────────────────────────

def configure_logging(level: str = "INFO") -> None:
    """Configure loguru with a consistent format."""
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{line}</cyan> – {message}",
        colorize=True,
    )


# ── Path helpers ─────────────────────────────────────────────────────────────

def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def ensure_dirs(*dirs: str | Path) -> None:
    """Create directories if they do not exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


# ── DuckDB helpers ───────────────────────────────────────────────────────────

def get_duckdb_path() -> Path:
    """Return the path to the shared DuckDB database file."""
    root = project_root()
    db_path = root / "data" / "f1.duckdb"
    ensure_dirs(db_path.parent)
    return db_path


# ── MLflow helpers ────────────────────────────────────────────────────────────

def get_mlflow_tracking_uri() -> str:
    return os.getenv(
        "MLFLOW_TRACKING_URI",
        load_config().get("mlflow", {}).get("tracking_uri", "http://localhost:5000"),
    )
