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
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from train_race_points_model import build_model_dataset

OUTCOME_CLASSES = ["WIN", "PODIUM", "POINTS", "NO_POINTS", "DNF"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a multi-class race outcome model using FP/Q features with classes "
            "WIN, PODIUM, POINTS, NO_POINTS, DNF."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data") / "fastf1_csv")
    parser.add_argument("--train-start-year", type=int, default=2018)
    parser.add_argument("--train-end-year", type=int, default=2024)
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--tree-estimators", type=int, default=450)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--dnf-threshold",
        type=float,
        default=0.2,
        help="Probability threshold used to predict DNF instead of best non-DNF class.",
    )
    parser.add_argument(
        "--dnf-threshold-mode",
        choices=["fixed", "auto-recall-floor", "auto-dnf-f1", "auto-macro-f1", "auto-balanced"],
        default="auto-recall-floor",
        help="How to choose the DNF threshold. Auto modes tune on a validation year.",
    )
    parser.add_argument(
        "--validation-year",
        type=int,
        default=None,
        help="Validation year for threshold tuning. Defaults to train-end-year.",
    )
    parser.add_argument(
        "--min-dnf-recall",
        type=float,
        default=0.1,
        help="Recall floor used by auto-recall-floor threshold mode.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models") / "race_outcome_multiclass.joblib",
    )
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=Path("data") / "model_outputs" / "race_outcome_multiclass_predictions.csv",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("data") / "model_outputs" / "race_outcome_multiclass_metrics.json",
    )
    return parser.parse_args()


def make_outcome_class(dataset: pd.DataFrame) -> pd.Series:
    dnf = dataset["dnf_target"].fillna(0).astype(int) == 1
    winner = dataset["winner_target"].fillna(0).astype(int) == 1
    podium = dataset["podium_target"].fillna(0).astype(int) == 1
    points = dataset["points_target"].fillna(0).astype(int) == 1

    classes = np.where(
        dnf,
        "DNF",
        np.where(winner, "WIN", np.where(podium, "PODIUM", np.where(points, "POINTS", "NO_POINTS"))),
    )
    return pd.Series(classes, index=dataset.index, dtype="object")


def event_topk_hit_rate(predictions: pd.DataFrame, label_name: str, k: int) -> float:
    hits: list[float] = []
    for _, group in predictions.groupby(["year", "round", "event_name"]):
        pred_topk = set(group.nlargest(k, "win_probability")["driver"].tolist())
        truth = set(group.loc[group["actual_outcome"] == label_name, "driver"].tolist())
        if not truth:
            continue
        hits.append(float(len(pred_topk & truth) / len(truth)))
    return float(np.mean(hits)) if hits else float("nan")


def build_model(
    args: argparse.Namespace,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> Pipeline:
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

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", ensemble)])


def decode_outcomes(
    proba_map: dict[str, np.ndarray],
    dnf_threshold: float,
) -> np.ndarray:
    n_rows = len(next(iter(proba_map.values()))) if proba_map else 0
    prob_dnf = proba_map.get("DNF", np.zeros(n_rows, dtype=float))

    finish_labels = [label for label in OUTCOME_CLASSES if label != "DNF"]
    finish_scores = np.column_stack(
        [proba_map.get(label, np.zeros(n_rows, dtype=float)) for label in finish_labels]
    )
    finish_idx = np.argmax(finish_scores, axis=1) if n_rows else np.array([], dtype=int)
    best_finish = np.array([finish_labels[idx] for idx in finish_idx], dtype=object) if n_rows else np.array([], dtype=object)

    return np.where(prob_dnf >= dnf_threshold, "DNF", best_finish)


def choose_dnf_threshold(
    y_true: pd.Series,
    proba_map: dict[str, np.ndarray],
    mode: str,
    fallback_threshold: float,
    min_dnf_recall: float,
) -> tuple[float, dict[str, Any]]:
    if mode == "fixed":
        return float(fallback_threshold), {"mode": "fixed"}

    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    y_true = y_true.astype(str)
    true_dnf = (y_true == "DNF").astype(int).to_numpy()

    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        pred = decode_outcomes(proba_map, float(threshold))
        pred_dnf = (pred == "DNF").astype(int)
        tp = int(((pred_dnf == 1) & (true_dnf == 1)).sum())
        fp = int(((pred_dnf == 1) & (true_dnf == 0)).sum())
        fn = int(((pred_dnf == 0) & (true_dnf == 1)).sum())

        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        dnf_f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0

        rows.append(
            {
                "threshold": float(threshold),
                "dnf_precision": precision,
                "dnf_recall": recall,
                "dnf_f1": dnf_f1,
                "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
            }
        )

    if not rows:
        return float(fallback_threshold), {"mode": "fixed", "fallback_reason": "no-threshold-grid"}

    if mode == "auto-recall-floor":
        eligible = [row for row in rows if row["dnf_recall"] >= min_dnf_recall]
        pool = eligible if eligible else rows
        selected = sorted(pool, key=lambda row: (row["macro_f1"], row["dnf_f1"]), reverse=True)[0]
    elif mode == "auto-dnf-f1":
        selected = sorted(rows, key=lambda row: (row["dnf_f1"], row["macro_f1"]), reverse=True)[0]
    elif mode == "auto-macro-f1":
        selected = sorted(rows, key=lambda row: (row["macro_f1"], row["dnf_f1"]), reverse=True)[0]
    elif mode == "auto-balanced":
        selected = sorted(rows, key=lambda row: (row["balanced_accuracy"], row["dnf_f1"]), reverse=True)[0]
    else:
        selected = {"threshold": float(fallback_threshold)}

    details: dict[str, Any] = {
        "mode": mode,
        "search_min": float(thresholds.min()),
        "search_max": float(thresholds.max()),
        "search_step": 0.01,
        "min_dnf_recall": float(min_dnf_recall),
    }
    details.update({k: float(v) for k, v in selected.items() if k != "threshold"})
    return float(selected["threshold"]), details


def train_and_evaluate(dataset: pd.DataFrame, args: argparse.Namespace) -> tuple[Pipeline, pd.DataFrame, dict[str, Any]]:
    dataset = dataset.copy()
    dataset["outcome_class"] = make_outcome_class(dataset)

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
        "points_target",
        "podium_target",
        "winner_target",
        "dnf_target",
        "accident_target",
        "mechanical_target",
        "other_retirement_target",
        "outcome_class",
    }

    numeric_cols = [
        c
        for c in dataset.columns
        if c not in non_feature_cols and c not in categorical_cols and pd.api.types.is_numeric_dtype(dataset[c])
    ]
    feature_cols = numeric_cols + categorical_cols

    X_train = dataset.loc[train_mask, feature_cols]
    y_train = dataset.loc[train_mask, "outcome_class"].astype(str)
    X_test = dataset.loc[test_mask, feature_cols]
    y_test = dataset.loc[test_mask, "outcome_class"].astype(str)

    dnf_threshold = float(args.dnf_threshold)
    threshold_details: dict[str, Any] = {"mode": "fixed"}
    validation_year = args.validation_year if args.validation_year is not None else args.train_end_year

    if args.dnf_threshold_mode != "fixed":
        tune_train_mask = train_mask & (dataset["year"] < validation_year)
        tune_val_mask = train_mask & (dataset["year"] == validation_year)

        if int(tune_train_mask.sum()) > 0 and int(tune_val_mask.sum()) > 0:
            tune_model = build_model(args, numeric_cols, categorical_cols)
            X_tune_train = dataset.loc[tune_train_mask, feature_cols]
            y_tune_train = dataset.loc[tune_train_mask, "outcome_class"].astype(str)
            X_tune_val = dataset.loc[tune_val_mask, feature_cols]
            y_tune_val = dataset.loc[tune_val_mask, "outcome_class"].astype(str)

            tune_model.fit(X_tune_train, y_tune_train)
            tune_proba = tune_model.predict_proba(X_tune_val)
            tune_class_order = list(tune_model.classes_)
            tune_proba_map = {label: tune_proba[:, idx] for idx, label in enumerate(tune_class_order)}

            dnf_threshold, threshold_details = choose_dnf_threshold(
                y_true=y_tune_val,
                proba_map=tune_proba_map,
                mode=args.dnf_threshold_mode,
                fallback_threshold=dnf_threshold,
                min_dnf_recall=args.min_dnf_recall,
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

    proba = model.predict_proba(X_test)
    class_order = list(model.classes_)
    proba_map = {label: proba[:, idx] for idx, label in enumerate(class_order)}
    pred = decode_outcomes(proba_map, dnf_threshold)

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
        "dnf_target",
        "outcome_class",
    ]
    predictions = dataset.loc[test_mask, output_cols].copy()
    predictions = predictions.rename(columns={"outcome_class": "actual_outcome"})
    predictions["predicted_outcome"] = pred

    for label in OUTCOME_CLASSES:
        predictions[f"prob_{label.lower()}"] = proba_map.get(label, np.zeros(len(predictions)))

    predictions["win_probability"] = predictions["prob_win"]

    metrics: dict[str, Any] = {
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "train_year_start": args.train_start_year,
        "train_year_end": args.train_end_year,
        "test_year": args.test_year,
        "dnf_threshold": float(dnf_threshold),
        "dnf_threshold_mode": args.dnf_threshold_mode,
        "dnf_threshold_details": threshold_details,
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        "win_top1_hit_rate": event_topk_hit_rate(predictions, "WIN", 1),
        "podium_top3_hit_rate": event_topk_hit_rate(predictions, "PODIUM", 3),
        "dnf_precision": float(
            (
                (predictions["actual_outcome"] == "DNF")
                & (predictions["predicted_outcome"] == "DNF")
            ).sum()
            / max((predictions["predicted_outcome"] == "DNF").sum(), 1)
        ),
        "dnf_recall": float(
            (
                (predictions["actual_outcome"] == "DNF")
                & (predictions["predicted_outcome"] == "DNF")
            ).sum()
            / max((predictions["actual_outcome"] == "DNF").sum(), 1)
        ),
        "class_distribution_actual": {
            k: int(v) for k, v in predictions["actual_outcome"].value_counts().to_dict().items()
        },
        "class_distribution_predicted": {
            k: int(v) for k, v in predictions["predicted_outcome"].value_counts().to_dict().items()
        },
    }

    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    metrics["classification_report"] = report

    labels = sorted(set(y_test) | set(pred))
    cm = confusion_matrix(y_test, pred, labels=labels)
    metrics["confusion_matrix_labels"] = labels
    metrics["confusion_matrix"] = cm.tolist()

    return model, predictions, metrics


def additional_data_needed() -> list[str]:
    return [
        "Official FIA pre-race starting grid after all penalties and pit-lane starts.",
        "Race-day weather windows and track evolution features per stint segment.",
        "Safety-car / VSC historical propensity by circuit and weekend format.",
        "Team pit-stop execution quality and strategy response timing features.",
        "Power-unit and reliability state (component age / replacement history).",
        "Fine-grained tyre degradation context (fuel-corrected long-run pace curves).",
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
    for key in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "win_top1_hit_rate",
        "podium_top3_hit_rate",
        "dnf_precision",
        "dnf_recall",
    ]:
        print(f"- {key}: {metrics[key]}")

    print("\nAdditional data that would improve race outcome quality:")
    for item in additional_data_needed():
        print(f"- {item}")


if __name__ == "__main__":
    main()
