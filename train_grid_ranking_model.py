from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from train_grid_position_model import build_driver_event_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a LightGBM ranking model for per-race grid position ordering."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data") / "fastf1_csv")
    parser.add_argument("--train-start-year", type=int, default=2018)
    parser.add_argument("--train-end-year", type=int, default=2024)
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models") / "grid_position_ranker.joblib",
    )
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=Path("data") / "model_outputs" / "grid_ranking_predictions.csv",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("data") / "model_outputs" / "grid_ranking_metrics.json",
    )
    return parser.parse_args()


def add_relative_features(dataset: pd.DataFrame) -> pd.DataFrame:
    df = dataset.copy()
    event_cols = ["year", "round", "event_name"]

    candidate_cols = [
        "fp1_best_lap_seconds",
        "fp2_best_lap_seconds",
        "fp3_best_lap_seconds",
        "fp1_long_run_median_seconds",
        "fp2_long_run_median_seconds",
        "fp3_long_run_median_seconds",
    ]

    for col in candidate_cols:
        if col not in df.columns:
            continue
        event_min = df.groupby(event_cols)[col].transform("min")
        team_min = df.groupby(event_cols + ["team"])[col].transform("min")
        df[f"{col}_delta_event_min"] = df[col] - event_min
        df[f"{col}_delta_team_min"] = df[col] - team_min

    return df


def build_relevance(df: pd.DataFrame) -> np.ndarray:
    # For ranking objective, higher relevance = better grid position.
    group_size = df.groupby(["year", "round", "event_name"])["grid_position_target"].transform("count")
    relevance = (group_size + 1 - df["grid_position_target"]).astype(float)
    return relevance.to_numpy()


def group_sizes(df: pd.DataFrame) -> list[int]:
    grp = df.groupby(["year", "round", "event_name"], sort=False).size()
    return grp.astype(int).tolist()


def mean_event_spearman(predictions: pd.DataFrame) -> float:
    vals: list[float] = []
    for _, group in predictions.groupby(["year", "round", "event_name"]):
        corr = group["actual_grid_position"].corr(group["predicted_grid_position"], method="spearman")
        if pd.notna(corr):
            vals.append(float(corr))
    return float(np.mean(vals)) if vals else float("nan")


def ndcg_at_k_by_event(predictions: pd.DataFrame, k: int) -> float:
    event_scores: list[float] = []
    for _, group in predictions.groupby(["year", "round", "event_name"]):
        group = group.sort_values("predicted_grid_score", ascending=False).reset_index(drop=True)
        gains = np.maximum(0.0, group["relevance"].to_numpy())

        dcg = 0.0
        for i, gain in enumerate(gains[:k], start=1):
            dcg += gain / np.log2(i + 1)

        ideal = np.sort(gains)[::-1]
        idcg = 0.0
        for i, gain in enumerate(ideal[:k], start=1):
            idcg += gain / np.log2(i + 1)

        if idcg > 0:
            event_scores.append(float(dcg / idcg))

    return float(np.mean(event_scores)) if event_scores else float("nan")


def train_and_evaluate(dataset: pd.DataFrame, args: argparse.Namespace) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    dataset = add_relative_features(dataset)
    dataset = dataset.sort_values(["year", "round", "event_name", "driver"]).reset_index(drop=True)

    train_mask = (dataset["year"] >= args.train_start_year) & (dataset["year"] <= args.train_end_year)
    test_mask = dataset["year"] == args.test_year

    if train_mask.sum() == 0:
        raise RuntimeError("No training rows found. Adjust train year range.")
    if test_mask.sum() == 0:
        raise RuntimeError("No test rows found. Adjust test year.")

    feature_cols = [
        c
        for c in dataset.columns
        if c.startswith("fp1_")
        or c.startswith("fp2_")
        or c.startswith("fp3_")
        or c.endswith("_delta_event_min")
        or c.endswith("_delta_team_min")
    ] + ["round", "driver", "driver_number", "team", "event_name"]
    feature_cols = [c for c in feature_cols if c in dataset.columns]
    feature_cols = list(dict.fromkeys(feature_cols))

    categorical_cols = [c for c in ["driver", "driver_number", "team", "event_name"] if c in feature_cols]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_cols,
            ),
            (
                "cat",
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

    base_cols = ["year", "round", "event_name", "driver", "grid_position_target"]
    selected_cols = list(dict.fromkeys(base_cols + feature_cols))

    train_df = dataset.loc[train_mask, selected_cols].copy()
    test_df = dataset.loc[test_mask, selected_cols].copy()

    X_train = preprocessor.fit_transform(train_df[feature_cols])
    X_test = preprocessor.transform(test_df[feature_cols])

    y_train = build_relevance(train_df)
    y_test = build_relevance(test_df)

    train_groups = group_sizes(train_df)
    test_groups = group_sizes(test_df)

    ranker = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=63,
        min_child_samples=10,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=args.random_state,
    )

    ranker.fit(
        X_train,
        y_train,
        group=train_groups,
        eval_set=[(X_test, y_test)],
        eval_group=[test_groups],
        eval_at=[3, 10],
    )

    scores = ranker.predict(X_test)

    predictions = test_df[["year", "round", "event_name", "driver", "team", "grid_position_target"]].copy()
    predictions = predictions.rename(columns={"grid_position_target": "actual_grid_position"})
    predictions["predicted_grid_score"] = scores
    predictions["relevance"] = y_test

    predictions["predicted_grid_position"] = (
        predictions.groupby(["year", "round", "event_name"])["predicted_grid_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    winners = predictions.loc[predictions["actual_grid_position"] == 1]

    metrics: dict[str, Any] = {
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "train_year_start": args.train_start_year,
        "train_year_end": args.train_end_year,
        "test_year": args.test_year,
        "mae_ranked_position": float(
            mean_absolute_error(predictions["actual_grid_position"], predictions["predicted_grid_position"])
        ),
        "winner_accuracy": float((winners["predicted_grid_position"] == 1).mean()) if len(winners) else float("nan"),
        "mean_event_spearman": mean_event_spearman(predictions),
        "ndcg_at_3": ndcg_at_k_by_event(predictions, k=3),
        "ndcg_at_10": ndcg_at_k_by_event(predictions, k=10),
    }

    model_bundle = {
        "preprocessor": preprocessor,
        "ranker": ranker,
        "feature_cols": feature_cols,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "metadata": {
            "objective": "lambdarank",
            "eval_at": [3, 10],
            "train_year_start": args.train_start_year,
            "train_year_end": args.train_end_year,
            "test_year": args.test_year,
        },
    }

    return model_bundle, predictions, metrics


def additional_data_needed() -> list[str]:
    return [
        "True FIA post-penalty starting-grid labels (current target is qualifying-lap based).",
        "Session weather + track evolution to model one-lap pace volatility.",
        "Explicit sprint-weekend format flags and sprint outcome features.",
        "Driver/team rolling form and circuit-specialization features.",
        "Reliability features (DNF risk, PU age, incident-prone circuit markers).",
    ]


def main() -> None:
    args = parse_args()

    dataset = build_driver_event_dataset(args.data_root)
    model_bundle, predictions, metrics = train_and_evaluate(dataset, args)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_bundle, args.model_out)
    predictions.to_csv(args.predictions_out, index=False)

    payload = {
        "metrics": metrics,
        "additional_data_needed": additional_data_needed(),
    }
    with args.metrics_out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print("Ranking training complete.")
    print(f"Model saved to: {args.model_out}")
    print(f"Predictions saved to: {args.predictions_out}")
    print(f"Metrics saved to: {args.metrics_out}")
    print("\nCore ranking metrics:")
    for key, value in metrics.items():
        print(f"- {key}: {value}")

    print("\nAdditional data that would improve ranking quality:")
    for item in additional_data_needed():
        print(f"- {item}")


if __name__ == "__main__":
    main()
