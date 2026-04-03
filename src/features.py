"""
src/features.py – Feature engineering: transforms raw Parquet files into the
driver–race dataset used for training and inference.

All features are computed from strictly pre-race information to avoid data
leakage.
"""
from __future__ import annotations

from typing import Any

import duckdb
import numpy as np
import pandas as pd
from loguru import logger

from src.utils import (
    configure_logging,
    ensure_dirs,
    get_duckdb_path,
    load_config,
    project_root,
)

configure_logging()

PROCESSED_DIR = project_root() / "data" / "processed"
SNAPSHOTS_DIR = project_root() / "data" / "snapshots"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_parquet(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found: {path}. Run src/ingest.py first."
        )
    return pd.read_parquet(path)


# ── Feature builders ─────────────────────────────────────────────────────────

def build_driver_form(
    results: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """
    Compute rolling driver form features over the last *window* races (excluding
    the current one).

    Returns a DataFrame keyed by (season, round, driver_id).
    """
    results = results.sort_values(["driver_id", "season", "round"]).copy()
    results["is_dnf"] = (results["finish_position"] >= 21).astype(int)

    rows: list[dict[str, Any]] = []
    for driver_id, grp in results.groupby("driver_id"):
        grp = grp.reset_index(drop=True)
        for i, row in grp.iterrows():
            past = grp.iloc[max(0, i - window) : i]  # type: ignore[misc]
            rows.append(
                {
                    "season": row["season"],
                    "round": row["round"],
                    "driver_id": driver_id,
                    "form_avg_finish": past["finish_position"].mean()
                    if len(past)
                    else np.nan,
                    "form_avg_points": past["points"].mean() if len(past) else 0.0,
                    "form_dnf_rate": past["is_dnf"].mean() if len(past) else 0.0,
                    "form_races_counted": len(past),
                }
            )

    return pd.DataFrame(rows)


def build_constructor_strength(
    results: pd.DataFrame,
    constructor_standings: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge constructor standing position and season points for each driver–race.
    """
    # Use the season-level standings (round 0 = end of season, or max round)
    season_standings = (
        constructor_standings.sort_values("round", ascending=False)
        .groupby(["season", "constructor_id"])
        .first()
        .reset_index()
        .rename(
            columns={
                "constructor_standings_pos": "constructor_standings_pos",
                "constructor_points": "constructor_pts_season",
            }
        )
    )
    merged = results.merge(
        season_standings[
            [
                "season",
                "constructor_id",
                "constructor_standings_pos",
                "constructor_pts_season",
            ]
        ],
        on=["season", "constructor_id"],
        how="left",
    )
    return merged[
        [
            "season",
            "round",
            "driver_id",
            "constructor_id",
            "constructor_standings_pos",
            "constructor_pts_season",
        ]
    ]


def build_circuit_history(
    results: pd.DataFrame,
    window: int = 3,
) -> pd.DataFrame:
    """
    Compute each driver's average finishing position at this circuit over the
    last *window* visits (strictly before the current race date).
    """
    results = results.sort_values(["driver_id", "circuit_id", "season", "round"])

    rows: list[dict[str, Any]] = []
    for (driver_id, circuit_id), grp in results.groupby(["driver_id", "circuit_id"]):
        grp = grp.reset_index(drop=True)
        for i, row in grp.iterrows():
            past = grp.iloc[max(0, i - window) : i]  # type: ignore[misc]
            rows.append(
                {
                    "season": row["season"],
                    "round": row["round"],
                    "driver_id": driver_id,
                    "circuit_id": circuit_id,
                    "circuit_avg_finish": past["finish_position"].mean()
                    if len(past)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


# ── Main feature pipeline ────────────────────────────────────────────────────

def build_feature_dataset(
    results: pd.DataFrame | None = None,
    qualifying: pd.DataFrame | None = None,
    constructor_standings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build the full driver–race feature dataset.

    If DataFrames are not supplied they are loaded from data/processed/.
    Returns a DataFrame with one row per driver–race and saves a snapshot.
    """
    cfg = load_config()
    window = cfg["features"]["form_window"]
    circuit_window = cfg["features"]["circuit_history_window"]

    if results is None:
        results = _load_parquet("results")
    if qualifying is None:
        qualifying = _load_parquet("qualifying")
    if constructor_standings is None:
        constructor_standings = _load_parquet("constructor_standings")

    logger.info("Building driver form features…")
    form_df = build_driver_form(results, window=window)

    logger.info("Building constructor strength features…")
    constructor_df = build_constructor_strength(results, constructor_standings)

    logger.info("Building circuit history features…")
    circuit_df = build_circuit_history(results, window=circuit_window)

    # ── Base dataset: one row per driver–race ──
    base = results[
        [
            "season",
            "round",
            "race_name",
            "circuit_id",
            "date",
            "driver_id",
            "driver_code",
            "constructor_id",
            "grid",
            "finish_position",
            "points",
            "status",
        ]
    ].copy()

    # Merge qualifying
    qual_cols = ["season", "round", "driver_id", "qualifying_position"]
    base = base.merge(
        qualifying[qual_cols],
        on=["season", "round", "driver_id"],
        how="left",
    )

    # Merge driver form
    base = base.merge(
        form_df,
        on=["season", "round", "driver_id"],
        how="left",
    )

    # Merge constructor strength
    base = base.merge(
        constructor_df.drop(columns=["constructor_id"], errors="ignore"),
        on=["season", "round", "driver_id"],
        how="left",
    )

    # Merge circuit history
    base = base.merge(
        circuit_df,
        on=["season", "round", "driver_id", "circuit_id"],
        how="left",
    )

    # ── Derived features ──
    base["round_fraction"] = base["round"] / base.groupby("season")["round"].transform(
        "max"
    )
    base["is_wet"] = 0  # TODO: join actual weather data from FastF1 or weather API
    base["circuit_id_enc"] = base["circuit_id"].astype("category").cat.codes

    # Fill remaining NaNs with sensible defaults
    base["qualifying_position"] = base["qualifying_position"].fillna(
        base["qualifying_position"].median()
    )
    base["form_avg_finish"] = base["form_avg_finish"].fillna(10.5)
    base["form_avg_points"] = base["form_avg_points"].fillna(0.0)
    base["form_dnf_rate"] = base["form_dnf_rate"].fillna(0.0)
    base["circuit_avg_finish"] = base["circuit_avg_finish"].fillna(10.5)
    base["constructor_standings_pos"] = base["constructor_standings_pos"].fillna(10)
    base["constructor_pts_season"] = base["constructor_pts_season"].fillna(0.0)

    logger.info(f"Feature dataset: {base.shape} rows×cols")

    # ── Persist snapshot ──
    ensure_dirs(SNAPSHOTS_DIR)
    snapshot_path = SNAPSHOTS_DIR / "features_latest.parquet"
    base.to_parquet(snapshot_path, index=False)
    logger.info(f"Feature snapshot saved → {snapshot_path}")

    # ── Store in DuckDB for SQL querying ──
    _write_to_duckdb(base)

    return base


def _write_to_duckdb(df: pd.DataFrame) -> None:
    db_path = str(get_duckdb_path())
    con = duckdb.connect(db_path)
    con.execute("DROP TABLE IF EXISTS features")
    con.execute("CREATE TABLE features AS SELECT * FROM df")
    con.close()
    logger.info(f"Feature table written to DuckDB: {db_path}")


def load_feature_snapshot() -> pd.DataFrame:
    """Load the most recent feature snapshot from disk."""
    path = SNAPSHOTS_DIR / "features_latest.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature snapshot not found at {path}. Run build_feature_dataset() first."
        )
    return pd.read_parquet(path)


def build_inference_row(
    driver_id: str,
    constructor_id: str,
    grid: int,
    qualifying_position: int,
    circuit_id: str,
    season: int,
    round_num: int,
    total_rounds: int = 24,
    form_avg_finish: float = 10.5,
    form_avg_points: float = 5.0,
    form_dnf_rate: float = 0.05,
    constructor_standings_pos: int = 5,
    constructor_pts_season: float = 100.0,
    circuit_avg_finish: float = 10.5,
    is_wet: int = 0,
) -> pd.DataFrame:
    """
    Build a single-row inference DataFrame for a driver–race without historical
    data. All defaults represent a mid-field driver; supply actual values for
    better predictions.
    """
    # Encode circuit_id using the known category mapping stored in DuckDB
    try:
        con = duckdb.connect(str(get_duckdb_path()), read_only=True)
        result = con.execute(
            "SELECT circuit_id_enc FROM features WHERE circuit_id = ? LIMIT 1",
            [circuit_id],
        ).fetchone()
        con.close()
        circuit_id_enc = result[0] if result else -1
    except Exception:
        circuit_id_enc = -1

    return pd.DataFrame(
        [
            {
                "season": season,
                "round": round_num,
                "circuit_id": circuit_id,
                "driver_id": driver_id,
                "constructor_id": constructor_id,
                "grid": grid,
                "qualifying_position": qualifying_position,
                "form_avg_finish": form_avg_finish,
                "form_avg_points": form_avg_points,
                "form_dnf_rate": form_dnf_rate,
                "form_races_counted": 5,
                "constructor_standings_pos": constructor_standings_pos,
                "constructor_pts_season": constructor_pts_season,
                "circuit_avg_finish": circuit_avg_finish,
                "round_fraction": round_num / total_rounds,
                "is_wet": is_wet,
                "circuit_id_enc": circuit_id_enc,
            }
        ]
    )


if __name__ == "__main__":
    build_feature_dataset()
