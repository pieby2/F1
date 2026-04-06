from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from train_grid_position_model import build_driver_event_dataset

RACE_TARGET_COLS = [
    "year",
    "round",
    "event_name",
    "driver_number",
    "driver_code",
    "driver",
    "team",
    "position",
    "classified_position",
    "status",
    "status_category",
    "points",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an ensemble classifier that predicts whether a driver will finish "
            "in the points (top-10) for a race weekend."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data") / "fastf1_csv")
    parser.add_argument("--train-start-year", type=int, default=2018)
    parser.add_argument("--train-end-year", type=int, default=2024)
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--validation-year",
        type=int,
        default=2024,
        help=(
            "Year used to tune blending between model probability and grid-prior probability. "
            "Set outside train range to disable tuning."
        ),
    )
    parser.add_argument(
        "--blend-step",
        type=float,
        default=0.05,
        help="Alpha step for blend search in [0,1].",
    )
    parser.add_argument(
        "--grid-prior-strength",
        type=float,
        default=10.0,
        help="Laplace-style smoothing strength for grid position prior.",
    )
    parser.add_argument(
        "--min-val-roc-improvement",
        type=float,
        default=0.002,
        help=(
            "Minimum validation ROC-AUC gain over pure grid prior required to choose a non-zero blend alpha."
        ),
    )
    parser.add_argument("--tree-estimators", type=int, default=400)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models") / "race_points_classifier.joblib",
    )
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=Path("data") / "model_outputs" / "race_points_predictions.csv",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("data") / "model_outputs" / "race_points_metrics.json",
    )
    return parser.parse_args()


def load_race_targets(data_root: Path) -> pd.DataFrame:
    race_files = sorted((data_root / "race_results").rglob("*.csv"))
    if not race_files:
        raise RuntimeError("No race result CSV files found.")

    frames: list[pd.DataFrame] = []
    for race_file in race_files:
        frame = pd.read_csv(race_file, usecols=RACE_TARGET_COLS)
        if frame.empty:
            continue
        frames.append(frame)

    race = pd.concat(frames, ignore_index=True)
    # Feature rows use three-letter driver codes from lap filenames.
    race["driver"] = race["driver_code"].fillna(race["driver"]).astype(str)
    race["driver_number"] = race["driver_number"].astype(str)
    race["position"] = pd.to_numeric(race["position"], errors="coerce")
    race["points"] = pd.to_numeric(race["points"], errors="coerce").fillna(0.0)

    race["points_target"] = (race["points"] > 0).astype(int)
    race["podium_target"] = race["position"].between(1, 3, inclusive="both").fillna(False).astype(int)
    race["winner_target"] = (race["position"] == 1).fillna(False).astype(int)

    key = ["year", "round", "event_name", "driver", "driver_number"]
    race = race.sort_values(key).drop_duplicates(subset=key, keep="first")
    return race


def add_context_features(dataset: pd.DataFrame) -> pd.DataFrame:
    event_key = ["year", "round", "event_name"]

    relative_sources = [
        "fp1_best_lap_seconds",
        "fp2_best_lap_seconds",
        "fp3_best_lap_seconds",
        "fp1_long_run_median_seconds",
        "fp2_long_run_median_seconds",
        "fp3_long_run_median_seconds",
        "q_best_lap_seconds",
    ]

    for col in relative_sources:
        if col not in dataset.columns:
            continue
        dataset[f"{col}_delta_event_min"] = dataset[col] - dataset.groupby(event_key)[col].transform("min")
        dataset[f"{col}_delta_team_min"] = dataset[col] - dataset.groupby(event_key + ["team"])[col].transform("min")

    classified = dataset["classified_position"].astype(str).str.strip().str.upper()
    dataset["dnf_target"] = (~classified.str.fullmatch(r"\d+")).astype(int)
    status_norm = dataset["status_category"].astype(str).str.strip().str.lower()
    dataset["accident_target"] = (status_norm == "accident").astype(int)
    dataset["mechanical_target"] = (status_norm == "mechanical failure").astype(int)
    dataset["other_retirement_target"] = (
        (dataset["dnf_target"] == 1)
        & (dataset["accident_target"] == 0)
        & (dataset["mechanical_target"] == 0)
    ).astype(int)

    dataset = dataset.sort_values(["driver", "year", "round"]).reset_index(drop=True)
    dataset["driver_points_rate_last5"] = dataset.groupby("driver")["points_target"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    dataset["driver_podium_rate_last5"] = dataset.groupby("driver")["podium_target"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    dataset["driver_avg_grid_last5"] = dataset.groupby("driver")["grid_position_target"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    dataset["driver_dnf_rate_last5"] = dataset.groupby("driver")["dnf_target"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    dataset["driver_accident_rate_last8"] = dataset.groupby("driver")["accident_target"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=1).mean()
    )
    dataset["driver_mechanical_rate_last8"] = dataset.groupby("driver")["mechanical_target"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=1).mean()
    )
    dataset["driver_other_retirement_rate_last8"] = dataset.groupby("driver")["other_retirement_target"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=1).mean()
    )
    dataset["driver_dnf_count_last3"] = dataset.groupby("driver")["dnf_target"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).sum()
    )

    dataset = dataset.sort_values(["team", "year", "round"]).reset_index(drop=True)
    dataset["team_points_rate_last5"] = dataset.groupby("team")["points_target"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    dataset["team_avg_grid_last5"] = dataset.groupby("team")["grid_position_target"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    dataset["team_dnf_rate_last5"] = dataset.groupby("team")["dnf_target"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    dataset["team_accident_rate_last8"] = dataset.groupby("team")["accident_target"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=1).mean()
    )
    dataset["team_mechanical_rate_last8"] = dataset.groupby("team")["mechanical_target"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=1).mean()
    )
    dataset["team_other_retirement_rate_last8"] = dataset.groupby("team")["other_retirement_target"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=1).mean()
    )
    dataset["team_dnf_count_last3"] = dataset.groupby("team")["dnf_target"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).sum()
    )

    dataset = dataset.sort_values(["driver", "event_name", "year", "round"]).reset_index(drop=True)
    dataset["driver_event_points_rate_hist"] = dataset.groupby(["driver", "event_name"])["points_target"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    dataset["driver_event_dnf_rate_hist"] = dataset.groupby(["driver", "event_name"])["dnf_target"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    dataset["driver_event_accident_rate_hist"] = dataset.groupby(["driver", "event_name"])["accident_target"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    dataset["driver_event_mechanical_rate_hist"] = dataset.groupby(["driver", "event_name"])["mechanical_target"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )

    dataset = dataset.sort_values(["team", "event_name", "year", "round"]).reset_index(drop=True)
    dataset["team_event_points_rate_hist"] = dataset.groupby(["team", "event_name"])["points_target"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    dataset["team_event_dnf_rate_hist"] = dataset.groupby(["team", "event_name"])["dnf_target"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    dataset["team_event_accident_rate_hist"] = dataset.groupby(["team", "event_name"])["accident_target"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    dataset["team_event_mechanical_rate_hist"] = dataset.groupby(["team", "event_name"])["mechanical_target"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )

    dataset["year_progress"] = dataset["round"] / dataset.groupby("year")["round"].transform("max")

    # --- Weather-aware performance features ---
    if "wx_rain_any_session" in dataset.columns:
        # Track whether this is a wet weekend
        rain_flag = dataset["wx_rain_any_session"].fillna(0).astype(int)

        # Driver wet performance: points rate in previous rainy races
        dataset = dataset.sort_values(["driver", "year", "round"]).reset_index(drop=True)
        wet_races = dataset.loc[rain_flag == 1].copy()
        if not wet_races.empty and "points_target" in wet_races.columns:
            wet_rates = wet_races.groupby("driver")["points_target"].transform(
                lambda s: s.shift(1).expanding(min_periods=1).mean()
            )
            dataset.loc[rain_flag == 1, "driver_wet_points_rate_hist"] = wet_rates
        if "driver_wet_points_rate_hist" not in dataset.columns:
            dataset["driver_wet_points_rate_hist"] = pd.NA

        # Team wet performance
        dataset = dataset.sort_values(["team", "year", "round"]).reset_index(drop=True)
        wet_races_team = dataset.loc[rain_flag == 1].copy()
        if not wet_races_team.empty and "points_target" in wet_races_team.columns:
            wet_team_rates = wet_races_team.groupby("team")["points_target"].transform(
                lambda s: s.shift(1).expanding(min_periods=1).mean()
            )
            dataset.loc[rain_flag == 1, "team_wet_points_rate_hist"] = wet_team_rates
        if "team_wet_points_rate_hist" not in dataset.columns:
            dataset["team_wet_points_rate_hist"] = pd.NA

    # --- Sprint position delta (race-day form indicator) ---
    if "sprint_position" in dataset.columns and "sprint_grid_position" in dataset.columns:
        dataset["sprint_position_delta"] = (
            dataset["sprint_position"] - dataset["sprint_grid_position"]
        )

    # --- Official grid position as feature ---
    if "official_grid_position" in dataset.columns:
        dataset["official_grid_position"] = pd.to_numeric(
            dataset["official_grid_position"], errors="coerce"
        )

    return dataset.sort_values(["year", "round", "event_name", "driver"]).reset_index(drop=True)


def build_model_dataset(data_root: Path) -> pd.DataFrame:
    features = build_driver_event_dataset(data_root)
    targets = load_race_targets(data_root)

    key = ["year", "round", "event_name", "driver", "driver_number"]
    dataset = features.merge(
        targets[
            key
            + [
                "team",
                "position",
                "classified_position",
                "status",
                "status_category",
                "points",
                "points_target",
                "podium_target",
                "winner_target",
            ]
        ],
        on=key,
        how="inner",
        suffixes=("", "_race"),
    )

    if dataset.empty:
        raise RuntimeError("No overlapping feature rows and race target rows were found.")

    if "team_race" in dataset.columns:
        dataset["team"] = dataset["team_race"].fillna(dataset["team"])
        dataset = dataset.drop(columns=["team_race"])

    dataset["team"] = dataset["team"].fillna("UNKNOWN")
    dataset = add_context_features(dataset)
    return dataset


def event_precision_at_10(predictions: pd.DataFrame) -> float:
    vals: list[float] = []
    for _, group in predictions.groupby(["year", "round", "event_name"]):
        top10 = group.nlargest(10, "points_probability")
        vals.append(float(top10["points_target"].mean()))
    return float(np.mean(vals)) if vals else float("nan")


def event_overlap_at_10(predictions: pd.DataFrame) -> float:
    overlaps: list[float] = []
    for _, group in predictions.groupby(["year", "round", "event_name"]):
        pred_top10 = set(group.nlargest(10, "points_probability")["driver"].tolist())
        actual_points = set(group.loc[group["points_target"] == 1, "driver"].tolist())
        denom = max(len(actual_points), 1)
        overlaps.append(float(len(pred_top10 & actual_points) / denom))
    return float(np.mean(overlaps)) if overlaps else float("nan")


def build_grid_prior(
    frame: pd.DataFrame,
    grid_col: str = "grid_position_target",
    target_col: str = "points_target",
    strength: float = 10.0,
) -> tuple[dict[int, float], float]:
    if frame.empty:
        return {}, 0.5

    global_rate = float(frame[target_col].mean())
    grouped = frame.groupby(grid_col)[target_col].agg(["mean", "count"]).reset_index()

    prior_map: dict[int, float] = {}
    for _, row in grouped.iterrows():
        grid = int(row[grid_col])
        mean = float(row["mean"])
        count = float(row["count"])
        smoothed = (mean * count + global_rate * strength) / (count + strength)
        prior_map[grid] = float(smoothed)

    return prior_map, global_rate


def apply_grid_prior(
    frame: pd.DataFrame,
    prior_map: dict[int, float],
    fallback: float,
    grid_col: str = "grid_position_target",
) -> np.ndarray:
    return (
        frame[grid_col]
        .map(lambda v: prior_map.get(int(v), fallback) if pd.notna(v) else fallback)
        .astype(float)
        .to_numpy()
    )


def tune_blend_alpha(
    y_true: pd.Series,
    model_proba: np.ndarray,
    grid_prior_proba: np.ndarray,
    meta_frame: pd.DataFrame,
    step: float,
    min_val_roc_improvement: float,
) -> tuple[float, float, float, float]:
    def safe_roc_auc(y: pd.Series, score: np.ndarray) -> float:
        y_arr = y.astype(int).to_numpy()
        if len(np.unique(y_arr)) < 2:
            return float("nan")
        return float(roc_auc_score(y_arr, score))

    candidates: list[dict[str, float]] = []

    if step <= 0:
        step = 0.05

    alpha = 0.0
    while alpha <= 1.000001:
        blended = alpha * model_proba + (1.0 - alpha) * grid_prior_proba
        pred = meta_frame.copy()
        pred["points_target"] = y_true.to_numpy()
        pred["points_probability"] = blended
        overlap = event_overlap_at_10(pred)
        roc = safe_roc_auc(y_true, blended)
        candidates.append(
            {
                "alpha": float(round(alpha, 6)),
                "overlap": float(overlap),
                "roc_auc": float(roc),
            }
        )
        alpha += step

    if not candidates:
        return 1.0, float("nan"), float("nan"), float("nan")

    best_overlap = max(item["overlap"] for item in candidates)
    overlap_candidates = [item for item in candidates if abs(item["overlap"] - best_overlap) < 1e-12]

    grid_candidate = min(candidates, key=lambda item: item["alpha"])
    grid_roc = grid_candidate["roc_auc"]

    if np.isnan(grid_roc):
        chosen = min(overlap_candidates, key=lambda item: item["alpha"])
        return chosen["alpha"], chosen["overlap"], chosen["roc_auc"], grid_roc

    improved_candidates = [
        item
        for item in overlap_candidates
        if (not np.isnan(item["roc_auc"])) and item["roc_auc"] >= grid_roc + min_val_roc_improvement
    ]

    if improved_candidates:
        chosen = min(improved_candidates, key=lambda item: item["alpha"])
    else:
        chosen = min(overlap_candidates, key=lambda item: item["alpha"])

    return chosen["alpha"], chosen["overlap"], chosen["roc_auc"], grid_roc


def train_and_evaluate(dataset: pd.DataFrame, args: argparse.Namespace) -> tuple[Pipeline, pd.DataFrame, dict[str, Any]]:
    train_mask = (dataset["year"] >= args.train_start_year) & (dataset["year"] <= args.train_end_year)
    test_mask = dataset["year"] == args.test_year
    val_mask = dataset["year"] == args.validation_year

    if train_mask.sum() == 0:
        raise RuntimeError("No training rows found. Adjust train year range.")
    if test_mask.sum() == 0:
        raise RuntimeError("No test rows found. Adjust test year.")

    categorical_cols = [c for c in ["driver", "driver_number", "team", "event_name"] if c in dataset.columns]
    non_feature_cols = {
        "position",
        "classified_position",
        "status",
        "status_category",
        "points",
        "points_target",
        "podium_target",
        "winner_target",
        "dnf_target",
        "accident_target",
        "mechanical_target",
        "other_retirement_target",
    }

    numeric_cols = [
        c
        for c in dataset.columns
        if c not in non_feature_cols and c not in categorical_cols and pd.api.types.is_numeric_dtype(dataset[c])
    ]
    feature_cols = numeric_cols + categorical_cols

    target_col = "points_target"

    X_train = dataset.loc[train_mask, feature_cols]
    y_train = dataset.loc[train_mask, target_col].astype(int)
    X_test = dataset.loc[test_mask, feature_cols]
    y_test = dataset.loc[test_mask, target_col].astype(int)

    categorical_cols = [c for c in categorical_cols if c in feature_cols]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
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

    ensemble = VotingClassifier(
        estimators=[
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=args.tree_estimators,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=args.random_state,
                    n_jobs=-1,
                ),
            ),
            (
                "et",
                ExtraTreesClassifier(
                    n_estimators=args.tree_estimators,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=args.random_state,
                    n_jobs=-1,
                ),
            ),
            (
                "gbr",
                GradientBoostingClassifier(random_state=args.random_state),
            ),
        ],
        voting="soft",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", ensemble),
        ]
    )

    blend_alpha = 1.0
    val_overlap_at_best_alpha = float("nan")
    val_roc_at_best_alpha = float("nan")
    val_grid_only_roc_auc = float("nan")

    can_tune = (
        args.train_start_year <= args.validation_year <= args.train_end_year
        and val_mask.sum() > 0
        and (train_mask & ~val_mask).sum() > 0
    )

    if can_tune:
        tune_train_mask = train_mask & ~val_mask
        X_tune_train = dataset.loc[tune_train_mask, feature_cols]
        y_tune_train = dataset.loc[tune_train_mask, target_col].astype(int)
        X_val = dataset.loc[val_mask, feature_cols]
        y_val = dataset.loc[val_mask, target_col].astype(int)

        tune_model = clone(model)
        tune_model.fit(X_tune_train, y_tune_train)
        val_model_proba = tune_model.predict_proba(X_val)[:, 1]

        prior_map_val, global_val = build_grid_prior(
            dataset.loc[tune_train_mask, ["grid_position_target", target_col]],
            strength=args.grid_prior_strength,
        )
        val_grid_prior_proba = apply_grid_prior(
            dataset.loc[val_mask],
            prior_map_val,
            global_val,
            grid_col="grid_position_target",
        )

        val_meta = dataset.loc[val_mask, ["year", "round", "event_name", "driver"]].copy()
        blend_alpha, val_overlap_at_best_alpha, val_roc_at_best_alpha, val_grid_only_roc_auc = tune_blend_alpha(
            y_true=y_val,
            model_proba=val_model_proba,
            grid_prior_proba=val_grid_prior_proba,
            meta_frame=val_meta,
            step=args.blend_step,
            min_val_roc_improvement=args.min_val_roc_improvement,
        )

    model.fit(X_train, y_train)
    model_proba = model.predict_proba(X_test)[:, 1]

    prior_map_full, global_full = build_grid_prior(
        dataset.loc[train_mask, ["grid_position_target", target_col]],
        strength=args.grid_prior_strength,
    )
    grid_prior_proba = apply_grid_prior(
        dataset.loc[test_mask],
        prior_map_full,
        global_full,
        grid_col="grid_position_target",
    )

    proba = blend_alpha * model_proba + (1.0 - blend_alpha) * grid_prior_proba
    pred = (proba >= args.threshold).astype(int)

    output_cols = [
        "year",
        "round",
        "event_name",
        "driver",
        "driver_number",
        "team",
        "position",
        "points",
        "points_target",
    ]
    predictions = dataset.loc[test_mask, output_cols].copy()
    predictions["model_points_probability"] = model_proba
    predictions["grid_prior_probability"] = grid_prior_proba
    predictions["blend_alpha"] = blend_alpha
    predictions["points_probability"] = proba
    predictions["predicted_points_target"] = pred
    predictions["predicted_points_rank"] = (
        predictions.groupby(["year", "round", "event_name"])["points_probability"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    metrics: dict[str, Any] = {
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "train_year_start": args.train_start_year,
        "train_year_end": args.train_end_year,
        "test_year": args.test_year,
        "threshold": args.threshold,
        "validation_year": args.validation_year,
        "blend_alpha": float(blend_alpha),
        "val_overlap_at_best_alpha": float(val_overlap_at_best_alpha),
        "val_roc_at_best_alpha": float(val_roc_at_best_alpha),
        "val_grid_only_roc_auc": float(val_grid_only_roc_auc),
        "grid_prior_strength": args.grid_prior_strength,
        "min_val_roc_improvement": args.min_val_roc_improvement,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "average_precision": float(average_precision_score(y_test, proba)),
        "brier": float(brier_score_loss(y_test, proba)),
        "model_only_roc_auc": float(roc_auc_score(y_test, model_proba)),
        "grid_prior_only_roc_auc": float(roc_auc_score(y_test, grid_prior_proba)),
        "event_precision_at_10": event_precision_at_10(predictions),
        "event_overlap_at_10": event_overlap_at_10(predictions),
    }

    return model, predictions, metrics


def additional_data_needed() -> list[str]:
    return [
        "Official FIA starting-grid labels after penalties (for cleaner pre-race context).",
        "Session weather and track evolution data (temperature, rain, wind, grip evolution).",
        "Circuit-level metadata (street/permanent, high-downforce, overtaking profile).",
        "Pit stop quality and strategy priors from previous races in season.",
        "DNF/reliability signals (PU age, team reliability trend, incident-prone circuits).",
        "Sprint-specific labels and weekend format flags to avoid mixed-session bias.",
    ]


def main() -> None:
    args = parse_args()
    dataset = build_model_dataset(args.data_root)

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

    print("\nAdditional data that would improve race outcome quality:")
    for item in additional_data_needed():
        print(f"- {item}")


if __name__ == "__main__":
    main()
