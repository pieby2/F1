from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
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
        description="Train an ensemble DNF classifier from FP/Q weekend features."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data") / "fastf1_csv")
    parser.add_argument("--train-start-year", type=int, default=2018)
    parser.add_argument("--train-end-year", type=int, default=2024)
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--threshold-mode",
        choices=["fixed", "auto-f1", "auto-event-mae", "auto-rate-match"],
        default="auto-f1",
        help="How to set the decision threshold. auto modes tune on a validation year.",
    )
    parser.add_argument(
        "--validation-year",
        type=int,
        default=None,
        help="Validation year for threshold tuning. Defaults to train-end-year.",
    )
    parser.add_argument("--tree-estimators", type=int, default=400)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models") / "dnf_classifier.joblib",
    )
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=Path("data") / "model_outputs" / "dnf_predictions.csv",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("data") / "model_outputs" / "dnf_metrics.json",
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
    race["driver"] = race["driver_code"].fillna(race["driver"]).astype(str)
    race["driver_number"] = race["driver_number"].astype(str)
    race["position"] = pd.to_numeric(race["position"], errors="coerce")
    race["points"] = pd.to_numeric(race["points"], errors="coerce").fillna(0.0)

    # Any non-numeric classified position (R, DNF, DSQ, etc.) is treated as non-finish.
    classified = race["classified_position"].astype(str).str.strip().str.upper()
    race["dnf_target"] = (~classified.str.fullmatch(r"\d+")).astype(int)

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

    status_norm = dataset["status_category"].astype(str).str.strip().str.lower()
    dataset["accident_target"] = (status_norm == "accident").astype(int)
    dataset["mechanical_target"] = (status_norm == "mechanical failure").astype(int)
    dataset["other_retirement_target"] = (
        (dataset["dnf_target"] == 1)
        & (dataset["accident_target"] == 0)
        & (dataset["mechanical_target"] == 0)
    ).astype(int)

    dataset = dataset.sort_values(["driver", "year", "round"]).reset_index(drop=True)
    dataset["driver_dnf_rate_last5"] = dataset.groupby("driver")["dnf_target"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    dataset["driver_avg_grid_last5"] = dataset.groupby("driver")["grid_position_target"].transform(
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
    dataset["team_dnf_rate_last5"] = dataset.groupby("team")["dnf_target"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    dataset["team_avg_grid_last5"] = dataset.groupby("team")["grid_position_target"].transform(
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
                "dnf_target",
            ]
        ],
        on=key,
        how="inner",
        suffixes=("", "_race"),
    )

    if dataset.empty:
        raise RuntimeError("No overlapping feature rows and DNF target rows were found.")

    if "team_race" in dataset.columns:
        dataset["team"] = dataset["team_race"].fillna(dataset["team"])
        dataset = dataset.drop(columns=["team_race"])

    dataset["team"] = dataset["team"].fillna("UNKNOWN")
    dataset = add_context_features(dataset)
    return dataset


def event_dnf_mae(predictions: pd.DataFrame) -> float:
    event = (
        predictions.groupby(["year", "round", "event_name"], as_index=False)
        .agg(actual_dnf=("dnf_target", "sum"), predicted_dnf=("dnf_probability", "sum"))
    )
    return float(np.mean(np.abs(event["actual_dnf"] - event["predicted_dnf"])))


def event_dnf_count_mae(predictions: pd.DataFrame) -> float:
    event = (
        predictions.groupby(["year", "round", "event_name"], as_index=False)
        .agg(actual_dnf=("dnf_target", "sum"), predicted_dnf=("predicted_dnf_target", "sum"))
    )
    return float(np.mean(np.abs(event["actual_dnf"] - event["predicted_dnf"])))


def build_model(args: argparse.Namespace, numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
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

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", ensemble),
        ]
    )


def choose_threshold(
    y_true: pd.Series,
    proba: np.ndarray,
    mode: str,
    frame: pd.DataFrame,
    fallback_threshold: float,
) -> tuple[float, dict[str, Any]]:
    if mode == "fixed":
        return fallback_threshold, {"mode": "fixed"}

    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    best_threshold = fallback_threshold
    best_score = -np.inf
    best_extra: dict[str, Any] = {}

    for threshold in thresholds:
        pred = (proba >= threshold).astype(int)
        if mode == "auto-f1":
            score = f1_score(y_true, pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
                best_extra = {
                    "best_f1": float(score),
                    "best_precision": float(precision_score(y_true, pred, zero_division=0)),
                    "best_recall": float(recall_score(y_true, pred, zero_division=0)),
                }
        elif mode == "auto-event-mae":
            tune = frame.copy()
            tune["dnf_probability"] = proba
            tune["predicted_dnf_target"] = pred
            mae = event_dnf_count_mae(tune)
            score = -mae
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
                best_extra = {"best_event_dnf_count_mae": float(mae)}
        elif mode == "auto-rate-match":
            actual_rate = float(np.mean(y_true))
            predicted_rate = float(np.mean(pred))
            score = -abs(predicted_rate - actual_rate)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
                best_extra = {
                    "validation_dnf_rate": actual_rate,
                    "predicted_dnf_rate_at_threshold": predicted_rate,
                }

    details: dict[str, Any] = {
        "mode": mode,
        "search_min": float(thresholds.min()),
        "search_max": float(thresholds.max()),
        "search_step": 0.01,
    }
    details.update(best_extra)
    return best_threshold, details


def train_and_evaluate(dataset: pd.DataFrame, args: argparse.Namespace) -> tuple[Pipeline, pd.DataFrame, dict[str, Any]]:
    train_mask = (dataset["year"] >= args.train_start_year) & (dataset["year"] <= args.train_end_year)
    test_mask = dataset["year"] == args.test_year

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

    target_col = "dnf_target"

    X_train = dataset.loc[train_mask, feature_cols]
    y_train = dataset.loc[train_mask, target_col].astype(int)
    X_test = dataset.loc[test_mask, feature_cols]
    y_test = dataset.loc[test_mask, target_col].astype(int)

    categorical_cols = [c for c in categorical_cols if c in feature_cols]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    validation_year = args.validation_year if args.validation_year is not None else args.train_end_year
    threshold = float(args.threshold)
    threshold_details: dict[str, Any] = {"mode": "fixed"}

    if args.threshold_mode != "fixed":
        tune_train_mask = (dataset["year"] >= args.train_start_year) & (dataset["year"] <= args.train_end_year)
        tune_train_mask &= dataset["year"] < validation_year
        tune_val_mask = dataset["year"] == validation_year

        if int(tune_train_mask.sum()) > 0 and int(tune_val_mask.sum()) > 0:
            tune_model = build_model(args, numeric_cols, categorical_cols)
            X_tune_train = dataset.loc[tune_train_mask, feature_cols]
            y_tune_train = dataset.loc[tune_train_mask, target_col].astype(int)
            X_tune_val = dataset.loc[tune_val_mask, feature_cols]
            y_tune_val = dataset.loc[tune_val_mask, target_col].astype(int)

            tune_model.fit(X_tune_train, y_tune_train)
            tune_proba = tune_model.predict_proba(X_tune_val)[:, 1]

            tune_frame_cols = ["year", "round", "event_name", "dnf_target"]
            tune_frame = dataset.loc[tune_val_mask, tune_frame_cols].copy()

            threshold, threshold_details = choose_threshold(
                y_true=y_tune_val,
                proba=tune_proba,
                mode=args.threshold_mode,
                frame=tune_frame,
                fallback_threshold=threshold,
            )
            threshold_details["validation_year"] = int(validation_year)
            threshold_details["validation_rows"] = int(tune_val_mask.sum())
            threshold_details["tuning_train_rows"] = int(tune_train_mask.sum())
        else:
            threshold_details = {
                "mode": "fixed",
                "fallback_reason": "insufficient rows for threshold tuning",
                "validation_year": int(validation_year),
            }

    model = build_model(args, numeric_cols, categorical_cols)

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)

    output_cols = [
        "year",
        "round",
        "event_name",
        "driver",
        "driver_number",
        "team",
        "position",
        "classified_position",
        "status",
        "status_category",
        "dnf_target",
    ]
    predictions = dataset.loc[test_mask, output_cols].copy()
    predictions["dnf_probability"] = proba
    predictions["predicted_dnf_target"] = pred

    neg_label = 0
    metrics: dict[str, Any] = {
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "train_year_start": args.train_start_year,
        "train_year_end": args.train_end_year,
        "test_year": args.test_year,
        "threshold": float(threshold),
        "threshold_mode": args.threshold_mode,
        "threshold_details": threshold_details,
        "test_dnf_rate": float(y_test.mean()),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "finish_recall": float(recall_score(y_test, pred, pos_label=neg_label, zero_division=0)),
        "predicted_dnf_rate": float(np.mean(pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "average_precision": float(average_precision_score(y_test, proba)),
        "brier": float(brier_score_loss(y_test, proba)),
        "event_dnf_mae": event_dnf_mae(predictions),
        "event_dnf_count_mae": event_dnf_count_mae(predictions),
    }

    return model, predictions, metrics


def additional_data_needed() -> list[str]:
    return [
        "Power-unit element age and replacement history per driver-weekend.",
        "Historical incident and safety-car intensity by circuit/weekend type.",
        "Pit stop reliability and garage incident history at team-level.",
        "Weather volatility features (rain onset probability, gusty wind periods).",
        "Historical DNF type split (accident vs mechanical) as separate targets.",
        "Weekend format flags (sprint vs conventional) with parc ferme change impact.",
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

    print("\nAdditional data that would improve DNF modeling quality:")
    for item in additional_data_needed():
        print(f"- {item}")


if __name__ == "__main__":
    main()
