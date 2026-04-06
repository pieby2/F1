from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SESSION_ALIAS = {
    "Practice_1": "fp1",
    "Practice_2": "fp2",
    "Practice_3": "fp3",
    "Qualifying": "q",
    "Sprint": "sprint",
}

FP_ALIASES = ("fp1", "fp2", "fp3")

LAP_USECOLS = [
    "year",
    "round",
    "event_name",
    "driver",
    "driver_number",
    "lap_number",
    "lap_time_seconds",
    "sector_1_seconds",
    "sector_2_seconds",
    "sector_3_seconds",
    "speed_i1_kph",
    "speed_i2_kph",
    "speed_fl_kph",
    "speed_st_kph",
    "compound",
    "tyre_life",
    "track_status",
]

RACE_RESULT_USECOLS = ["year", "round", "driver_code", "team", "grid_position"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an ensemble model to predict qualifying grid position from "
            "FP1-FP3 lap-level CSV data."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data") / "fastf1_csv")
    parser.add_argument("--train-start-year", type=int, default=2018)
    parser.add_argument("--train-end-year", type=int, default=2024)
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models") / "grid_position_ensemble.joblib",
    )
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=Path("data") / "model_outputs" / "grid_position_predictions.csv",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("data") / "model_outputs" / "grid_position_metrics.json",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--tree-estimators", type=int, default=400)
    return parser.parse_args()


def safe_stat(series: pd.Series, fn: str) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return float("nan")
    if fn == "min":
        return float(numeric.min())
    if fn == "max":
        return float(numeric.max())
    if fn == "mean":
        return float(numeric.mean())
    if fn == "median":
        return float(numeric.median())
    if fn == "std":
        value = float(numeric.std(ddof=0))
        return 0.0 if math.isnan(value) else value
    raise ValueError(f"Unsupported stat function: {fn}")


def summarize_driver_session_file(csv_path: Path) -> dict[str, Any] | None:
    session_dir = csv_path.parent.name
    session_alias = SESSION_ALIAS.get(session_dir)
    if session_alias is None:
        return None

    frame = pd.read_csv(csv_path, usecols=LAP_USECOLS)
    if frame.empty:
        return None

    meta = frame.iloc[0]
    compounds = frame["compound"].astype("string").str.upper()

    long_run = frame.loc[
        pd.to_numeric(frame["tyre_life"], errors="coerce").between(5, 15, inclusive="both"),
        "lap_time_seconds",
    ]

    total_rows = max(len(frame), 1)
    soft_share = float((compounds == "SOFT").sum() / total_rows)
    medium_share = float((compounds == "MEDIUM").sum() / total_rows)
    hard_share = float((compounds == "HARD").sum() / total_rows)

    summary: dict[str, Any] = {
        "year": int(meta["year"]),
        "round": int(meta["round"]),
        "event_name": str(meta["event_name"]),
        "driver": str(meta["driver"]),
        "driver_number": str(meta["driver_number"]),
        "session": session_alias,
        "lap_count": int(pd.to_numeric(frame["lap_number"], errors="coerce").notna().sum()),
        "best_lap_seconds": safe_stat(frame["lap_time_seconds"], "min"),
        "median_lap_seconds": safe_stat(frame["lap_time_seconds"], "median"),
        "std_lap_seconds": safe_stat(frame["lap_time_seconds"], "std"),
        "best_sector_1_seconds": safe_stat(frame["sector_1_seconds"], "min"),
        "best_sector_2_seconds": safe_stat(frame["sector_2_seconds"], "min"),
        "best_sector_3_seconds": safe_stat(frame["sector_3_seconds"], "min"),
        "max_speed_i1_kph": safe_stat(frame["speed_i1_kph"], "max"),
        "max_speed_i2_kph": safe_stat(frame["speed_i2_kph"], "max"),
        "max_speed_fl_kph": safe_stat(frame["speed_fl_kph"], "max"),
        "max_speed_st_kph": safe_stat(frame["speed_st_kph"], "max"),
        "mean_track_status": safe_stat(frame["track_status"], "mean"),
        "compound_variety": int(compounds.nunique(dropna=True)),
        "soft_share": soft_share,
        "medium_share": medium_share,
        "hard_share": hard_share,
        "long_run_median_seconds": safe_stat(long_run, "median"),
    }
    return summary


# ---------------------------------------------------------------------------
# Weather feature loading
# ---------------------------------------------------------------------------

def load_weather_features(data_root: Path) -> pd.DataFrame:
    """Load weather summary CSVs and pivot into per-event per-session features."""
    weather_root = data_root / "weather"
    if not weather_root.exists():
        return pd.DataFrame()

    summary_files = sorted(weather_root.rglob("*_summary.csv"))
    if not summary_files:
        return pd.DataFrame()

    frames = []
    for f in summary_files:
        try:
            df = pd.read_csv(f)
            if not df.empty:
                frames.append(df)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    weather = pd.concat(frames, ignore_index=True)
    weather["year"] = pd.to_numeric(weather["year"], errors="coerce")
    weather["round"] = pd.to_numeric(weather["round"], errors="coerce")

    # Map session names to aliases
    session_map = {
        "Practice 1": "fp1", "Practice_1": "fp1", "FP1": "fp1",
        "Practice 2": "fp2", "Practice_2": "fp2", "FP2": "fp2",
        "Practice 3": "fp3", "Practice_3": "fp3", "FP3": "fp3",
        "Qualifying": "q", "Q": "q",
        "Sprint": "sprint", "S": "sprint",
        "Race": "race", "R": "race",
    }
    weather["session_alias"] = weather["session_name"].map(session_map)
    weather = weather.dropna(subset=["session_alias"])

    # Pivot weather features by session
    metric_cols = [
        "air_temp_mean", "air_temp_max", "air_temp_min",
        "track_temp_mean", "track_temp_max", "track_temp_min", "track_temp_std",
        "humidity_mean", "pressure_mean",
        "wind_speed_mean", "wind_speed_max", "wind_direction_mean",
        "rainfall_fraction", "temp_delta_mean",
    ]

    key_cols = ["year", "round", "event_name"]
    pivoted = None

    for alias in ["fp1", "fp2", "fp3", "q", "sprint", "race"]:
        subset = weather.loc[weather["session_alias"] == alias, key_cols + metric_cols].copy()
        if subset.empty:
            continue
        subset = subset.rename(columns={col: f"wx_{alias}_{col}" for col in metric_cols})
        subset = subset.drop_duplicates(subset=key_cols, keep="first")
        if pivoted is None:
            pivoted = subset
        else:
            pivoted = pivoted.merge(subset, on=key_cols, how="outer")

    if pivoted is None or pivoted.empty:
        return pd.DataFrame()

    # Add derived cross-session weather features
    rain_cols = [c for c in pivoted.columns if c.endswith("_rainfall_fraction")]
    if rain_cols:
        pivoted["wx_rain_any_session"] = (pivoted[rain_cols].max(axis=1) > 0).astype(int)
        pivoted["wx_max_rainfall_fraction"] = pivoted[rain_cols].max(axis=1)

    track_temp_cols = [c for c in pivoted.columns if "track_temp_mean" in c]
    if len(track_temp_cols) > 1:
        pivoted["wx_track_temp_variability"] = pivoted[track_temp_cols].std(axis=1, ddof=0)

    return pivoted


# ---------------------------------------------------------------------------
# Circuit feature loading
# ---------------------------------------------------------------------------

def load_circuit_features(data_root: Path) -> pd.DataFrame:
    """Load circuit info and merge into event-level features."""
    circuit_dir = data_root / "circuits"
    if not circuit_dir.exists():
        return pd.DataFrame()

    circuit_files = sorted(circuit_dir.glob("*.csv"))
    circuit_files = [f for f in circuit_files if "_corners" not in f.name]
    if not circuit_files:
        return pd.DataFrame()

    frames = []
    for f in circuit_files:
        try:
            df = pd.read_csv(f)
            if not df.empty:
                frames.append(df)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    circuits = pd.concat(frames, ignore_index=True)
    circuits = circuits.drop_duplicates(subset=["event_name"], keep="last")

    # Rename columns to avoid clashes
    rename_map = {}
    for col in circuits.columns:
        if col != "event_name":
            rename_map[col] = f"circuit_{col}"
    circuits = circuits.rename(columns=rename_map)

    return circuits


# ---------------------------------------------------------------------------
# Sprint feature loading
# ---------------------------------------------------------------------------

def load_sprint_features(data_root: Path) -> pd.DataFrame:
    """Load sprint result CSVs into per-driver features."""
    sprint_root = data_root / "sprint_results"
    if not sprint_root.exists():
        return pd.DataFrame()

    files = sorted(sprint_root.rglob("*.csv"))
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if not df.empty:
                frames.append(df)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    sprint = pd.concat(frames, ignore_index=True)
    sprint["driver"] = sprint["driver_code"].fillna(sprint.get("driver", pd.NA)).astype(str)
    sprint["driver_number"] = sprint["driver_number"].astype(str)
    sprint["sprint_position"] = pd.to_numeric(sprint["position"], errors="coerce")
    sprint["sprint_points"] = pd.to_numeric(sprint["points"], errors="coerce").fillna(0.0)
    sprint["sprint_grid_position"] = pd.to_numeric(sprint["grid_position"], errors="coerce")

    key = ["year", "round", "event_name", "driver", "driver_number"]
    result_cols = ["sprint_position", "sprint_points", "sprint_grid_position"]
    sprint = sprint[key + result_cols].drop_duplicates(subset=key, keep="first")
    sprint["has_sprint"] = 1

    return sprint


# ---------------------------------------------------------------------------
# Official grid position loading
# ---------------------------------------------------------------------------

def load_official_grid(data_root: Path) -> pd.DataFrame:
    """Load official grid positions from race results (includes penalties)."""
    race_root = data_root / "race_results"
    if not race_root.exists():
        return pd.DataFrame()

    files = sorted(race_root.rglob("*.csv"))
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            cols_available = pd.read_csv(f, nrows=0).columns.tolist()
            usecols = ["year", "round", "driver_code", "grid_position"]
            usecols = [c for c in usecols if c in cols_available]
            if "grid_position" not in usecols:
                continue
            df = pd.read_csv(f, usecols=usecols)
            if not df.empty:
                frames.append(df)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    grid = pd.concat(frames, ignore_index=True)
    grid = grid.rename(columns={"driver_code": "driver"})
    grid["official_grid_position"] = pd.to_numeric(grid["grid_position"], errors="coerce")
    grid = grid.drop(columns=["grid_position"], errors="ignore")
    grid = grid.dropna(subset=["official_grid_position"])
    key = ["year", "round", "driver"]
    grid = grid.drop_duplicates(subset=key, keep="first")
    return grid


# ---------------------------------------------------------------------------
# Main dataset builder
# ---------------------------------------------------------------------------

def build_driver_event_dataset(data_root: Path) -> pd.DataFrame:
    lap_root = data_root / "laps"
    race_results_root = data_root / "race_results"

    lap_files = sorted(lap_root.rglob("*.csv"))
    if not lap_files:
        raise RuntimeError(f"No lap CSV files found under {lap_root}")

    rows: list[dict[str, Any]] = []
    for lap_file in lap_files:
        row = summarize_driver_session_file(lap_file)
        if row is not None:
            rows.append(row)

    session_df = pd.DataFrame(rows)
    key_cols = ["year", "round", "event_name", "driver", "driver_number"]

    # --- FP features (wide pivot) ---
    fp_wide: pd.DataFrame | None = None
    metric_cols = [c for c in session_df.columns if c not in key_cols + ["session"]]

    for alias in FP_ALIASES:
        subset = session_df.loc[session_df["session"] == alias, key_cols + metric_cols].copy()
        if subset.empty:
            continue
        subset = subset.rename(columns={col: f"{alias}_{col}" for col in metric_cols})
        subset = subset.drop_duplicates(subset=key_cols, keep="first")

        if fp_wide is None:
            fp_wide = subset
        else:
            fp_wide = fp_wide.merge(subset, on=key_cols, how="outer")

    if fp_wide is None or fp_wide.empty:
        raise RuntimeError("No FP1/FP2/FP3 session rows were found.")

    # --- Sprint features (wide pivot) ---
    sprint_lap_data = session_df.loc[session_df["session"] == "sprint", key_cols + metric_cols].copy()
    if not sprint_lap_data.empty:
        sprint_lap_data = sprint_lap_data.rename(columns={col: f"sprint_{col}" for col in metric_cols})
        sprint_lap_data = sprint_lap_data.drop_duplicates(subset=key_cols, keep="first")
        fp_wide = fp_wide.merge(sprint_lap_data, on=key_cols, how="left")

    # --- Qualifying target ---
    q_df = session_df.loc[session_df["session"] == "q", key_cols + ["best_lap_seconds"]].copy()
    q_df = q_df.rename(columns={"best_lap_seconds": "q_best_lap_seconds"})
    q_df = q_df.dropna(subset=["q_best_lap_seconds"])

    event_cols = ["year", "round", "event_name"]
    q_df["grid_position_target"] = (
        q_df.groupby(event_cols)["q_best_lap_seconds"].rank(method="first").astype(int)
    )

    dataset = fp_wide.merge(
        q_df[key_cols + ["q_best_lap_seconds", "grid_position_target"]],
        on=key_cols,
        how="inner",
    )

    # --- Team from race results ---
    team_frames: list[pd.DataFrame] = []
    for race_file in sorted(race_results_root.rglob("*.csv")):
        try:
            cols_available = pd.read_csv(race_file, nrows=0).columns.tolist()
            usecols = [c for c in RACE_RESULT_USECOLS if c in cols_available]
            frame = pd.read_csv(race_file, usecols=usecols)
        except Exception:
            continue
        if frame.empty:
            continue
        frame = frame.rename(columns={"driver_code": "driver"})
        team_frames.append(frame)

    if team_frames:
        team_map = pd.concat(team_frames, ignore_index=True)
        team_map = team_map.drop_duplicates(subset=["year", "round", "driver"], keep="first")
        merge_cols = ["year", "round", "driver"]
        extra_cols = [c for c in team_map.columns if c not in merge_cols]
        dataset = dataset.merge(team_map[merge_cols + extra_cols], on=merge_cols, how="left")
    else:
        dataset["team"] = pd.NA

    dataset["team"] = dataset["team"].fillna("UNKNOWN")

    # --- Official grid position ---
    official_grid = load_official_grid(data_root)
    if not official_grid.empty:
        dataset = dataset.merge(official_grid, on=["year", "round", "driver"], how="left")
        # Grid penalty delta: how much official grid differs from qualifying rank
        if "official_grid_position" in dataset.columns:
            dataset["grid_penalty_delta"] = (
                dataset["official_grid_position"] - dataset["grid_position_target"]
            )

    # --- Weather features ---
    weather = load_weather_features(data_root)
    if not weather.empty:
        dataset = dataset.merge(weather, on=["year", "round", "event_name"], how="left")

    # --- Circuit features ---
    circuits = load_circuit_features(data_root)
    if not circuits.empty:
        dataset = dataset.merge(circuits, on=["event_name"], how="left")

    # --- Sprint results features ---
    sprint_results = load_sprint_features(data_root)
    if not sprint_results.empty:
        dataset = dataset.merge(sprint_results, on=key_cols, how="left")
        dataset["has_sprint"] = dataset["has_sprint"].fillna(0).astype(int)
        if "sprint_position" in dataset.columns and "sprint_grid_position" in dataset.columns:
            dataset["sprint_position_delta"] = (
                dataset["sprint_position"] - dataset["sprint_grid_position"]
            )
    else:
        dataset["has_sprint"] = 0

    dataset = dataset.sort_values(["year", "round", "event_name", "driver"]).reset_index(drop=True)
    return dataset


def train_and_evaluate(dataset: pd.DataFrame, args: argparse.Namespace) -> tuple[Pipeline, pd.DataFrame, dict[str, Any]]:
    train_mask = (dataset["year"] >= args.train_start_year) & (dataset["year"] <= args.train_end_year)
    test_mask = dataset["year"] == args.test_year

    if train_mask.sum() == 0:
        raise RuntimeError("No training rows found. Adjust train year range.")
    if test_mask.sum() == 0:
        raise RuntimeError("No test rows found. Adjust test year.")

    feature_cols = [
        col
        for col in dataset.columns
        if col.startswith("fp1_") or col.startswith("fp2_") or col.startswith("fp3_")
        or col.startswith("wx_") or col.startswith("circuit_") or col.startswith("sprint_")
    ] + ["driver", "driver_number", "team", "event_name", "has_sprint"]

    # Add official grid features if available
    for extra in ["official_grid_position", "grid_penalty_delta"]:
        if extra in dataset.columns:
            feature_cols.append(extra)

    feature_cols = [c for c in feature_cols if c in dataset.columns]
    feature_cols = list(dict.fromkeys(feature_cols))

    target_col = "grid_position_target"

    X_train = dataset.loc[train_mask, feature_cols]
    y_train = dataset.loc[train_mask, target_col].astype(float)
    X_test = dataset.loc[test_mask, feature_cols]
    y_test = dataset.loc[test_mask, target_col].astype(float)

    categorical_cols = ["driver", "driver_number", "team", "event_name"]
    categorical_cols = [c for c in categorical_cols if c in feature_cols]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_cols,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    ensemble = VotingRegressor(
        estimators=[
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=args.tree_estimators,
                    min_samples_leaf=2,
                    random_state=args.random_state,
                    n_jobs=-1,
                ),
            ),
            (
                "et",
                ExtraTreesRegressor(
                    n_estimators=args.tree_estimators,
                    min_samples_leaf=2,
                    random_state=args.random_state,
                    n_jobs=-1,
                ),
            ),
            (
                "gbr",
                GradientBoostingRegressor(random_state=args.random_state),
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", ensemble),
        ]
    )

    model.fit(X_train, y_train)
    raw_predictions = model.predict(X_test)

    output_cols = ["year", "round", "event_name", "driver", "driver_number", "team", target_col]
    predictions = dataset.loc[test_mask, output_cols].copy()
    predictions = predictions.rename(columns={target_col: "actual_grid_position"})
    predictions["predicted_grid_score"] = raw_predictions

    event_cols = ["year", "round", "event_name"]
    predictions["predicted_grid_position"] = (
        predictions.groupby(event_cols)["predicted_grid_score"].rank(method="first").astype(int)
    )

    event_rank_corrs: list[float] = []
    for _, group in predictions.groupby(event_cols):
        corr = group["actual_grid_position"].corr(group["predicted_grid_position"], method="spearman")
        if pd.notna(corr):
            event_rank_corrs.append(float(corr))

    winners = predictions.loc[predictions["actual_grid_position"] == 1]

    metrics: dict[str, Any] = {
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "train_year_start": args.train_start_year,
        "train_year_end": args.train_end_year,
        "test_year": args.test_year,
        "mae_raw_score": float(mean_absolute_error(y_test, predictions["predicted_grid_score"])),
        "mae_ranked_position": float(
            mean_absolute_error(predictions["actual_grid_position"], predictions["predicted_grid_position"])
        ),
        "top3_hit_rate": float(
            ((predictions["actual_grid_position"] <= 3) & (predictions["predicted_grid_position"] <= 3)).mean()
        ),
        "winner_accuracy": float((winners["predicted_grid_position"] == 1).mean()) if len(winners) else float("nan"),
        "mean_event_spearman": float(np.mean(event_rank_corrs)) if event_rank_corrs else float("nan"),
    }

    return model, predictions, metrics


def additional_data_needed() -> list[str]:
    return [
        "Official FIA starting-grid classifications including penalties/grid drops (current target is a qualifying-lap proxy).",
        "Track metadata (layout type, altitude, asphalt grip, overtaking difficulty) for circuit-specific effects.",
        "Car setup and power-unit state indicators (new PU elements, setup changes, parc ferme constraints).",
        "Historical driver/team form features before each weekend (rolling pace/rank over previous events).",
        "Tyre strategy context from practice (long-run fuel-corrected pace rather than raw lap times).",
    ]


def main() -> None:
    args = parse_args()

    dataset = build_driver_event_dataset(args.data_root)
    model, predictions, metrics = train_and_evaluate(dataset, args)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, args.model_out)
    predictions.to_csv(args.predictions_out, index=False)

    payload = {
        "metrics": metrics,
        "additional_data_needed": additional_data_needed(),
    }
    with args.metrics_out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print("Training complete.")
    print(f"Model saved to: {args.model_out}")
    print(f"Predictions saved to: {args.predictions_out}")
    print(f"Metrics saved to: {args.metrics_out}")
    print("\nCore metrics:")
    for key, value in metrics.items():
        print(f"- {key}: {value}")

    print("\nAdditional data that would improve grid prediction quality:")
    for item in additional_data_needed():
        print(f"- {item}")


if __name__ == "__main__":
    main()
