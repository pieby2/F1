"""
src/ingest.py – Data ingestion from Jolpica / Ergast-compatible REST API.

Each function fetches one resource, saves it as Parquet under data/raw/, and
returns a pandas DataFrame.

TODO: Replace the stub URL constants with real endpoints if you migrate away
      from the public Jolpica mirror or add authentication via JOLPICA_API_KEY.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from loguru import logger

from src.utils import configure_logging, ensure_dirs, load_config, project_root

configure_logging()

# ── Constants ────────────────────────────────────────────────────────────────

# Jolpica is a community-maintained Ergast-compatible API (no key required for
# public endpoints as of 2025).
BASE_URL = "https://api.jolpi.ca/ergast/f1"
RAW_DIR = project_root() / "data" / "raw"
RATE_LIMIT_SLEEP = 0.25  # seconds between requests to respect rate limits


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """GET JSON from *url* with simple retry logic."""
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()  # type: ignore[return-value]
        except requests.RequestException as exc:
            logger.warning(f"Request failed (attempt {attempt+1}): {exc}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch {url} after 3 attempts")


def _save_parquet(df: pd.DataFrame, name: str) -> Path:
    ensure_dirs(RAW_DIR)
    path = RAW_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    logger.info(f"Saved {len(df)} rows → {path}")
    return path


# ── Public ingest functions ──────────────────────────────────────────────────

def fetch_race_results(season: int) -> pd.DataFrame:
    """Fetch all race results for a given *season* year."""
    logger.info(f"Fetching race results for {season}…")
    url = f"{BASE_URL}/{season}/results.json"
    data = _get_json(url, params={"limit": 1000})
    races = data["MRData"]["RaceTable"]["Races"]

    rows: list[dict[str, Any]] = []
    for race in races:
        round_no = int(race["round"])
        circuit_id = race["Circuit"]["circuitId"]
        race_name = race["raceName"]
        date = race.get("date", "")
        for result in race.get("Results", []):
            driver = result["Driver"]
            constructor = result["Constructor"]
            rows.append(
                {
                    "season": season,
                    "round": round_no,
                    "race_name": race_name,
                    "circuit_id": circuit_id,
                    "date": date,
                    "driver_id": driver["driverId"],
                    "driver_code": driver.get("code", ""),
                    "constructor_id": constructor["constructorId"],
                    "grid": int(result.get("grid", 0)),
                    "position": result.get("position"),
                    "position_text": result.get("positionText", ""),
                    "points": float(result.get("points", 0)),
                    "status": result.get("status", ""),
                    "laps": int(result.get("laps", 0)),
                }
            )
        time.sleep(RATE_LIMIT_SLEEP)

    df = pd.DataFrame(rows)
    # Normalise finishing position: DNF → config.dnf_position
    cfg = load_config()
    dnf_pos = cfg["features"]["dnf_position"]
    df["finish_position"] = pd.to_numeric(df["position"], errors="coerce").fillna(
        dnf_pos
    ).astype(int)
    _save_parquet(df, f"results_{season}")
    return df


def fetch_qualifying(season: int) -> pd.DataFrame:
    """Fetch qualifying results for a given *season*."""
    logger.info(f"Fetching qualifying for {season}…")
    url = f"{BASE_URL}/{season}/qualifying.json"
    data = _get_json(url, params={"limit": 1000})
    races = data["MRData"]["RaceTable"]["Races"]

    rows: list[dict[str, Any]] = []
    for race in races:
        round_no = int(race["round"])
        circuit_id = race["Circuit"]["circuitId"]
        for q in race.get("QualifyingResults", []):
            driver = q["Driver"]
            rows.append(
                {
                    "season": season,
                    "round": round_no,
                    "circuit_id": circuit_id,
                    "driver_id": driver["driverId"],
                    "qualifying_position": int(q.get("position", 0)),
                    "q1": q.get("Q1", ""),
                    "q2": q.get("Q2", ""),
                    "q3": q.get("Q3", ""),
                }
            )
        time.sleep(RATE_LIMIT_SLEEP)

    df = pd.DataFrame(rows)
    _save_parquet(df, f"qualifying_{season}")
    return df


def fetch_driver_standings(season: int) -> pd.DataFrame:
    """Fetch end-of-season driver standings (used as season-context features)."""
    logger.info(f"Fetching driver standings for {season}…")
    url = f"{BASE_URL}/{season}/driverStandings.json"
    data = _get_json(url, params={"limit": 100})
    standings_lists = data["MRData"]["StandingsTable"]["StandingsLists"]

    rows: list[dict[str, Any]] = []
    for sl in standings_lists:
        round_no = int(sl.get("round", 0))
        for s in sl.get("DriverStandings", []):
            driver = s["Driver"]
            rows.append(
                {
                    "season": season,
                    "round": round_no,
                    "driver_id": driver["driverId"],
                    "driver_points": float(s.get("points", 0)),
                    "driver_wins": int(s.get("wins", 0)),
                    "driver_standings_pos": int(s.get("position", 0)),
                }
            )
    df = pd.DataFrame(rows)
    _save_parquet(df, f"driver_standings_{season}")
    return df


def fetch_constructor_standings(season: int) -> pd.DataFrame:
    """Fetch constructor standings for a given *season*."""
    logger.info(f"Fetching constructor standings for {season}…")
    url = f"{BASE_URL}/{season}/constructorStandings.json"
    data = _get_json(url, params={"limit": 100})
    standings_lists = data["MRData"]["StandingsTable"]["StandingsLists"]

    rows: list[dict[str, Any]] = []
    for sl in standings_lists:
        round_no = int(sl.get("round", 0))
        for s in sl.get("ConstructorStandings", []):
            constructor = s["Constructor"]
            rows.append(
                {
                    "season": season,
                    "round": round_no,
                    "constructor_id": constructor["constructorId"],
                    "constructor_points": float(s.get("points", 0)),
                    "constructor_wins": int(s.get("wins", 0)),
                    "constructor_standings_pos": int(s.get("position", 0)),
                }
            )
    df = pd.DataFrame(rows)
    _save_parquet(df, f"constructor_standings_{season}")
    return df


def ingest_seasons(seasons: list[int] | None = None) -> dict[str, pd.DataFrame]:
    """Run full ingestion for a list of *seasons* (defaults to config)."""
    cfg = load_config()
    if seasons is None:
        seasons = cfg["data"]["train_seasons"] + [cfg["data"]["val_season"]]

    results: list[pd.DataFrame] = []
    qualifying: list[pd.DataFrame] = []
    driver_standings: list[pd.DataFrame] = []
    constructor_standings: list[pd.DataFrame] = []

    for season in seasons:
        results.append(fetch_race_results(season))
        qualifying.append(fetch_qualifying(season))
        driver_standings.append(fetch_driver_standings(season))
        constructor_standings.append(fetch_constructor_standings(season))

    combined = {
        "results": pd.concat(results, ignore_index=True),
        "qualifying": pd.concat(qualifying, ignore_index=True),
        "driver_standings": pd.concat(driver_standings, ignore_index=True),
        "constructor_standings": pd.concat(constructor_standings, ignore_index=True),
    }

    # Persist combined datasets
    ensure_dirs(project_root() / "data" / "processed")
    for name, df in combined.items():
        path = project_root() / "data" / "processed" / f"{name}.parquet"
        df.to_parquet(path, index=False)
        logger.info(f"Combined {name}: {len(df)} rows → {path}")

    return combined


if __name__ == "__main__":
    ingest_seasons()
