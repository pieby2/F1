from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import fastf1
import numpy as np
import pandas as pd
import requests
from fastf1.exceptions import DataNotLoadedError

SESSION_CODES = ["FP1", "FP2", "FP3", "Q", "S", "R"]
ACCIDENT_KEYWORDS = {
    "accident",
    "collision",
    "crash",
    "damage",
    "spun off",
    "spun",
}
MECHANICAL_KEYWORDS = {
    "engine",
    "gearbox",
    "hydraulic",
    "transmission",
    "mechanical",
    "brake",
    "power unit",
    "turbo",
    "electrical",
    "driveshaft",
    "fuel",
    "suspension",
    "water leak",
    "oil leak",
    "radiator",
    "puncture",
    "wheel",
    "throttle",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download FastF1 lap-level data for FP1-FP3/Qualifying/Sprint/Race and "
            "store per-driver-session CSV files plus race classification results, "
            "weather data, sprint results, and circuit info."
        )
    )
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument(
        "--start-round",
        type=int,
        default=1,
        help="Round number to start from within start-year.",
    )
    parser.add_argument(
        "--end-round",
        type=int,
        default=None,
        help="Optional round number cap within end-year.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data") / "fastf1_csv")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSV files instead of skipping them.",
    )
    return parser.parse_args()


def sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip()).strip("_")


def get_round_numbers(year: int) -> list[int]:
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    rounds = sorted({int(rnd) for rnd in schedule["RoundNumber"].dropna().tolist() if int(rnd) > 0})
    return rounds


def series_or_na(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(pd.NA, index=df.index, dtype="object")


def td_to_seconds(series: pd.Series) -> pd.Series:
    return pd.to_timedelta(series, errors="coerce").dt.total_seconds()


def categorize_status(raw_status: str) -> str:
    status = str(raw_status or "").strip().lower()
    if not status:
        return "unknown"

    if status.startswith("finished") or status.startswith("+"):
        return "finished"

    if any(keyword in status for keyword in ACCIDENT_KEYWORDS):
        return "accident"

    if any(keyword in status for keyword in MECHANICAL_KEYWORDS):
        return "mechanical failure"

    return "other"


def load_session(year: int, round_number: int, session_code: str):
    session = fastf1.get_session(year, round_number, session_code)
    session.load(laps=True, telemetry=False, weather=True, messages=False)
    _ = session.laps
    return session


def write_laps_csv(
    session,
    year: int,
    round_number: int,
    output_root: Path,
    overwrite: bool,
) -> int:
    try:
        laps = session.laps
    except DataNotLoadedError:
        return 0
    if laps is None or laps.empty or "Driver" not in laps.columns:
        return 0

    event_name = sanitize(str(session.event["EventName"]))
    session_code = sanitize(str(session.name))
    session_dir = (
        output_root
        / "laps"
        / f"{year}"
        / f"round_{round_number:02d}_{event_name}"
        / session_code
    )
    session_dir.mkdir(parents=True, exist_ok=True)

    driver_count = 0
    for driver in sorted(laps["Driver"].dropna().unique()):
        driver_laps = laps.pick_drivers(driver).copy()
        if driver_laps.empty:
            continue

        out_file = session_dir / f"{sanitize(str(driver))}.csv"
        if out_file.exists() and not overwrite:
            continue

        out = pd.DataFrame(index=driver_laps.index)
        out["year"] = year
        out["round"] = round_number
        out["event_name"] = str(session.event["EventName"])
        out["session_name"] = str(session.name)
        out["session_type"] = str(session.session_info.get("Type", ""))
        out["driver"] = series_or_na(driver_laps, "Driver")
        out["driver_number"] = series_or_na(driver_laps, "DriverNumber")

        # Core lap timing fields
        out["lap_number"] = series_or_na(driver_laps, "LapNumber")
        out["lap_time_seconds"] = td_to_seconds(series_or_na(driver_laps, "LapTime"))
        out["sector_1_seconds"] = td_to_seconds(series_or_na(driver_laps, "Sector1Time"))
        out["sector_2_seconds"] = td_to_seconds(series_or_na(driver_laps, "Sector2Time"))
        out["sector_3_seconds"] = td_to_seconds(series_or_na(driver_laps, "Sector3Time"))

        # Tyre and stint metadata
        out["compound"] = series_or_na(driver_laps, "Compound")
        out["stint"] = series_or_na(driver_laps, "Stint")
        out["tyre_life"] = series_or_na(driver_laps, "TyreLife")
        out["fresh_tyre"] = series_or_na(driver_laps, "FreshTyre")

        # Speed trap fields
        out["speed_i1_kph"] = series_or_na(driver_laps, "SpeedI1")
        out["speed_i2_kph"] = series_or_na(driver_laps, "SpeedI2")
        out["speed_fl_kph"] = series_or_na(driver_laps, "SpeedFL")
        out["speed_st_kph"] = series_or_na(driver_laps, "SpeedST")

        # Track status flags and quality markers
        out["track_status"] = series_or_na(driver_laps, "TrackStatus")
        out["is_personal_best"] = series_or_na(driver_laps, "IsPersonalBest")
        out["is_accurate"] = series_or_na(driver_laps, "IsAccurate")
        out["pit_in_time_seconds"] = td_to_seconds(series_or_na(driver_laps, "PitInTime"))
        out["pit_out_time_seconds"] = td_to_seconds(series_or_na(driver_laps, "PitOutTime"))

        out.to_csv(out_file, index=False)
        driver_count += 1

    return driver_count


# ---------------------------------------------------------------------------
# Weather data export
# ---------------------------------------------------------------------------

def write_weather_csv(
    session,
    year: int,
    round_number: int,
    output_root: Path,
    overwrite: bool,
) -> bool:
    """Write per-session aggregated weather data to CSV."""
    try:
        weather = session.weather_data
    except Exception:
        return False

    if weather is None or weather.empty:
        return False

    event_name = sanitize(str(session.event["EventName"]))
    session_code = sanitize(str(session.name))
    weather_dir = output_root / "weather" / f"{year}" / f"round_{round_number:02d}_{event_name}"
    weather_dir.mkdir(parents=True, exist_ok=True)
    out_file = weather_dir / f"{session_code}.csv"

    if out_file.exists() and not overwrite:
        return False

    out = pd.DataFrame()

    # Map FastF1 weather columns to our standardized names
    col_map = {
        "AirTemp": "air_temp_c",
        "TrackTemp": "track_temp_c",
        "Humidity": "humidity_pct",
        "Pressure": "pressure_mbar",
        "WindSpeed": "wind_speed_kph",
        "WindDirection": "wind_direction_deg",
        "Rainfall": "rainfall",
    }

    for src_col, dst_col in col_map.items():
        if src_col in weather.columns:
            out[dst_col] = weather[src_col]
        else:
            out[dst_col] = pd.NA

    # Add metadata
    out.insert(0, "year", year)
    out.insert(1, "round", round_number)
    out.insert(2, "event_name", str(session.event["EventName"]))
    out.insert(3, "session_name", str(session.name))

    out.to_csv(out_file, index=False)

    # Also write a summary row for easy feature building
    summary_file = weather_dir / f"{session_code}_summary.csv"
    summary = pd.DataFrame([{
        "year": year,
        "round": round_number,
        "event_name": str(session.event["EventName"]),
        "session_name": str(session.name),
        "air_temp_mean": _safe_mean(out, "air_temp_c"),
        "air_temp_max": _safe_max(out, "air_temp_c"),
        "air_temp_min": _safe_min(out, "air_temp_c"),
        "track_temp_mean": _safe_mean(out, "track_temp_c"),
        "track_temp_max": _safe_max(out, "track_temp_c"),
        "track_temp_min": _safe_min(out, "track_temp_c"),
        "track_temp_std": _safe_std(out, "track_temp_c"),
        "humidity_mean": _safe_mean(out, "humidity_pct"),
        "pressure_mean": _safe_mean(out, "pressure_mbar"),
        "wind_speed_mean": _safe_mean(out, "wind_speed_kph"),
        "wind_speed_max": _safe_max(out, "wind_speed_kph"),
        "wind_direction_mean": _safe_mean(out, "wind_direction_deg"),
        "rainfall_fraction": _rainfall_fraction(out),
        "temp_delta_mean": _safe_temp_delta(out),
    }])
    summary.to_csv(summary_file, index=False)

    return True


def _safe_mean(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return float("nan")
    vals = pd.to_numeric(df[col], errors="coerce")
    return float(vals.mean()) if vals.notna().any() else float("nan")


def _safe_max(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return float("nan")
    vals = pd.to_numeric(df[col], errors="coerce")
    return float(vals.max()) if vals.notna().any() else float("nan")


def _safe_min(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return float("nan")
    vals = pd.to_numeric(df[col], errors="coerce")
    return float(vals.min()) if vals.notna().any() else float("nan")


def _safe_std(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return float("nan")
    vals = pd.to_numeric(df[col], errors="coerce")
    return float(vals.std(ddof=0)) if vals.notna().sum() > 1 else 0.0


def _rainfall_fraction(df: pd.DataFrame) -> float:
    if "rainfall" not in df.columns:
        return 0.0
    rain = df["rainfall"]
    total = len(rain)
    if total == 0:
        return 0.0
    rain_bool = rain.astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes"})
    return float(rain_bool.sum() / total)


def _safe_temp_delta(df: pd.DataFrame) -> float:
    if "track_temp_c" not in df.columns or "air_temp_c" not in df.columns:
        return float("nan")
    track = pd.to_numeric(df["track_temp_c"], errors="coerce")
    air = pd.to_numeric(df["air_temp_c"], errors="coerce")
    delta = track - air
    return float(delta.mean()) if delta.notna().any() else float("nan")


# ---------------------------------------------------------------------------
# Race results (with official grid position)
# ---------------------------------------------------------------------------

def race_results_from_fastf1(race_session, year: int, round_number: int) -> pd.DataFrame:
    results = race_session.results
    if results is None or len(results) == 0:
        return pd.DataFrame()

    results = results.copy()
    out = pd.DataFrame(index=results.index)
    out["year"] = year
    out["round"] = round_number
    out["event_name"] = str(race_session.event["EventName"])
    out["driver_number"] = series_or_na(results, "DriverNumber")
    out["driver_code"] = series_or_na(results, "Abbreviation")
    out["driver"] = series_or_na(results, "FullName")
    out["team"] = series_or_na(results, "TeamName")
    out["position"] = series_or_na(results, "Position")
    out["classified_position"] = series_or_na(results, "ClassifiedPosition")
    out["grid_position"] = series_or_na(results, "GridPosition")  # Official grid with penalties
    out["status"] = series_or_na(results, "Status")
    out["status_category"] = out["status"].map(categorize_status)
    out["points"] = series_or_na(results, "Points")

    # Extract Q1/Q2/Q3 times if available
    out["q1_seconds"] = td_to_seconds(series_or_na(results, "Q1"))
    out["q2_seconds"] = td_to_seconds(series_or_na(results, "Q2"))
    out["q3_seconds"] = td_to_seconds(series_or_na(results, "Q3"))

    return out


def race_results_from_ergast(year: int, round_number: int) -> pd.DataFrame:
    url = f"https://ergast.com/api/f1/{year}/{round_number}/results.json"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        return pd.DataFrame()

    race = races[0]
    rows = []
    for item in race.get("Results", []):
        status = item.get("status", "")
        rows.append(
            {
                "year": year,
                "round": round_number,
                "event_name": race.get("raceName", ""),
                "driver_number": item.get("number"),
                "driver_code": item.get("Driver", {}).get("code"),
                "driver": (
                    f"{item.get('Driver', {}).get('givenName', '')} "
                    f"{item.get('Driver', {}).get('familyName', '')}"
                ).strip(),
                "team": item.get("Constructor", {}).get("name"),
                "position": item.get("position"),
                "classified_position": item.get("positionText"),
                "grid_position": item.get("grid"),
                "status": status,
                "status_category": categorize_status(status),
                "points": item.get("points"),
                "q1_seconds": pd.NA,
                "q2_seconds": pd.NA,
                "q3_seconds": pd.NA,
            }
        )

    return pd.DataFrame(rows)


def write_race_results_csv(
    race_session,
    year: int,
    round_number: int,
    output_root: Path,
    overwrite: bool,
) -> bool:
    event_name = sanitize(str(race_session.event["EventName"]))
    race_dir = output_root / "race_results" / f"{year}"
    race_dir.mkdir(parents=True, exist_ok=True)
    out_file = race_dir / f"round_{round_number:02d}_{event_name}.csv"

    if out_file.exists() and not overwrite:
        return False

    frame = race_results_from_fastf1(race_session, year, round_number)
    source = "fastf1"

    if frame.empty:
        frame = race_results_from_ergast(year, round_number)
        source = "ergast"

    if frame.empty:
        return False

    frame["source"] = source
    frame.to_csv(out_file, index=False)
    return True


# ---------------------------------------------------------------------------
# Sprint results
# ---------------------------------------------------------------------------

def write_sprint_results_csv(
    sprint_session,
    year: int,
    round_number: int,
    output_root: Path,
    overwrite: bool,
) -> bool:
    """Write sprint race classification results to CSV."""
    try:
        results = sprint_session.results
    except Exception:
        return False

    if results is None or len(results) == 0:
        return False

    event_name = sanitize(str(sprint_session.event["EventName"]))
    sprint_dir = output_root / "sprint_results" / f"{year}"
    sprint_dir.mkdir(parents=True, exist_ok=True)
    out_file = sprint_dir / f"round_{round_number:02d}_{event_name}.csv"

    if out_file.exists() and not overwrite:
        return False

    results = results.copy()
    out = pd.DataFrame(index=results.index)
    out["year"] = year
    out["round"] = round_number
    out["event_name"] = str(sprint_session.event["EventName"])
    out["driver_number"] = series_or_na(results, "DriverNumber")
    out["driver_code"] = series_or_na(results, "Abbreviation")
    out["driver"] = series_or_na(results, "FullName")
    out["team"] = series_or_na(results, "TeamName")
    out["position"] = series_or_na(results, "Position")
    out["classified_position"] = series_or_na(results, "ClassifiedPosition")
    out["grid_position"] = series_or_na(results, "GridPosition")
    out["status"] = series_or_na(results, "Status")
    out["status_category"] = out["status"].map(categorize_status)
    out["points"] = series_or_na(results, "Points")
    out["source"] = "fastf1"

    out.to_csv(out_file, index=False)
    return True


# ---------------------------------------------------------------------------
# Circuit info
# ---------------------------------------------------------------------------

def write_circuit_info(
    session,
    output_root: Path,
    overwrite: bool,
) -> bool:
    """Extract and save circuit metadata."""
    try:
        circuit_info = session.get_circuit_info()
    except Exception:
        return False

    if circuit_info is None:
        return False

    circuit_dir = output_root / "circuits"
    circuit_dir.mkdir(parents=True, exist_ok=True)

    # Use event name as circuit identifier
    event_name = sanitize(str(session.event["EventName"]))
    out_file = circuit_dir / f"{event_name}.csv"

    if out_file.exists() and not overwrite:
        return False

    # Extract corner info
    corners = getattr(circuit_info, "corners", None)
    num_corners = 0
    if corners is not None and not corners.empty:
        num_corners = len(corners)

    # Build summary
    summary = {
        "event_name": str(session.event["EventName"]),
        "circuit_key": str(getattr(session.event, "CircuitKey", event_name)),
        "num_corners": num_corners,
    }

    # Try to get rotation (track map orientation)
    rotation = getattr(circuit_info, "rotation", None)
    summary["track_rotation"] = float(rotation) if rotation is not None else float("nan")

    pd.DataFrame([summary]).to_csv(out_file, index=False)

    # Also save detailed corner data if available
    if corners is not None and not corners.empty:
        corners_file = circuit_dir / f"{event_name}_corners.csv"
        if not corners_file.exists() or overwrite:
            corners.to_csv(corners_file, index=False)

    return True


# ---------------------------------------------------------------------------
# Single-round ingestion (library-callable)
# ---------------------------------------------------------------------------

def ingest_single_round(
    year: int,
    round_number: int,
    output_dir: Path,
    overwrite: bool = False,
    session_codes: list[str] | None = None,
) -> dict[str, bool]:
    """Ingest data for a single round. Returns dict of what was written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "fastf1_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    codes = session_codes or SESSION_CODES
    result = {
        "laps": False,
        "weather": False,
        "race_results": False,
        "sprint_results": False,
        "circuit_info": False,
    }

    race_session = None
    sprint_session = None

    for session_code in codes:
        try:
            session = load_session(year, round_number, session_code)
        except Exception as exc:
            print(f"  [WARN] Skipping {year} round {round_number:02d} {session_code}: {exc}")
            continue

        if session_code == "R":
            race_session = session
        if session_code == "S":
            sprint_session = session

        # Write laps
        written = write_laps_csv(
            session=session,
            year=year,
            round_number=round_number,
            output_root=output_dir,
            overwrite=overwrite,
        )
        if written > 0:
            result["laps"] = True
        print(f"    [OK] {session_code} lap CSVs written for {written} drivers")

        # Write weather
        wx = write_weather_csv(
            session=session,
            year=year,
            round_number=round_number,
            output_root=output_dir,
            overwrite=overwrite,
        )
        if wx:
            result["weather"] = True
            print(f"    [OK] {session_code} weather CSV saved")

        # Circuit info (only need once per event)
        if not result["circuit_info"]:
            ci = write_circuit_info(
                session=session,
                output_root=output_dir,
                overwrite=overwrite,
            )
            if ci:
                result["circuit_info"] = True
                print(f"    [OK] Circuit info saved")

    # Race results
    if race_session is not None:
        try:
            saved = write_race_results_csv(
                race_session=race_session,
                year=year,
                round_number=round_number,
                output_root=output_dir,
                overwrite=overwrite,
            )
            result["race_results"] = saved
            print(f"    [OK] Race result CSV saved={saved}")
        except Exception as exc:
            print(f"    [WARN] Could not save race results for round {round_number:02d}: {exc}")

    # Sprint results
    if sprint_session is not None:
        try:
            saved = write_sprint_results_csv(
                sprint_session=sprint_session,
                year=year,
                round_number=round_number,
                output_root=output_dir,
                overwrite=overwrite,
            )
            result["sprint_results"] = saved
            print(f"    [OK] Sprint result CSV saved={saved}")
        except Exception as exc:
            print(f"    [WARN] Could not save sprint results for round {round_number:02d}: {exc}")

    return result


# ---------------------------------------------------------------------------
# Multi-year ingestion
# ---------------------------------------------------------------------------

def ingest_years(
    years: Iterable[int],
    output_dir: Path,
    overwrite: bool,
    start_year: int,
    end_year: int,
    start_round: int,
    end_round: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "fastf1_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    for year in years:
        print(f"=== Processing season {year} ===")
        try:
            rounds = get_round_numbers(year)
        except Exception as exc:
            print(f"[WARN] Could not load schedule for {year}: {exc}")
            continue

        if year == start_year:
            rounds = [rnd for rnd in rounds if rnd >= start_round]
        if year == end_year and end_round is not None:
            rounds = [rnd for rnd in rounds if rnd <= end_round]

        for round_number in rounds:
            print(f"  - Round {round_number:02d}")
            ingest_single_round(
                year=year,
                round_number=round_number,
                output_dir=output_dir,
                overwrite=overwrite,
            )


def main() -> None:
    args = parse_args()
    years = range(args.start_year, args.end_year + 1)
    ingest_years(
        years=years,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        start_year=args.start_year,
        end_year=args.end_year,
        start_round=args.start_round,
        end_round=args.end_round,
    )


if __name__ == "__main__":
    main()
