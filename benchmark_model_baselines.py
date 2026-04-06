from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from train_dnf_model import build_model_dataset as build_dnf_dataset
from train_grid_position_model import build_driver_event_dataset
from train_race_outcome_multiclass_model import make_outcome_class
from train_race_points_model import build_model_dataset as build_points_dataset

EVENT_KEY = ["year", "round", "event_name"]
ROW_KEY = EVENT_KEY + ["driver"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark model outputs against naive baselines for grid ranking, "
            "points classification, DNF classification, and multiclass race outcome."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data") / "fastf1_csv")
    parser.add_argument("--train-start-year", type=int, default=2018)
    parser.add_argument("--train-end-year", type=int, default=2024)
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument(
        "--grid-predictions",
        type=Path,
        default=Path("data") / "model_outputs" / "grid_ranking_predictions.csv",
    )
    parser.add_argument(
        "--points-predictions",
        type=Path,
        default=Path("data") / "model_outputs" / "race_points_predictions.csv",
    )
    parser.add_argument(
        "--dnf-predictions",
        type=Path,
        default=Path("data") / "model_outputs" / "dnf_predictions.csv",
    )
    parser.add_argument(
        "--outcome-predictions",
        type=Path,
        default=Path("data") / "model_outputs" / "race_outcome_multiclass_predictions.csv",
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        default=Path("data") / "model_outputs" / "model_baseline_benchmarks.json",
    )
    return parser.parse_args()


def safe_float(value: float | int | np.floating | np.integer | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def to_numeric(series: pd.Series, fallback: float) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(fallback)


def add_event_rank(df: pd.DataFrame, score_col: str, out_col: str) -> pd.DataFrame:
    ranked = df.copy()
    ranked[out_col] = (
        ranked.groupby(EVENT_KEY)[score_col].rank(method="first", ascending=False).astype(int)
    )
    return ranked


def mean_event_spearman(df: pd.DataFrame, actual_col: str, predicted_col: str) -> float:
    vals: list[float] = []
    for _, group in df.groupby(EVENT_KEY):
        corr = group[actual_col].corr(group[predicted_col], method="spearman")
        if pd.notna(corr):
            vals.append(float(corr))
    return float(np.mean(vals)) if vals else float("nan")


def ndcg_at_k(df: pd.DataFrame, score_col: str, actual_grid_col: str, k: int) -> float:
    scores: list[float] = []
    for _, group in df.groupby(EVENT_KEY):
        group = group.copy()
        group["relevance"] = group[actual_grid_col].count() + 1 - group[actual_grid_col]
        ranked = group.sort_values(score_col, ascending=False).reset_index(drop=True)
        gains = np.maximum(0.0, ranked["relevance"].to_numpy(dtype=float))

        dcg = 0.0
        for i, gain in enumerate(gains[:k], start=1):
            dcg += gain / np.log2(i + 1)

        ideal = np.sort(gains)[::-1]
        idcg = 0.0
        for i, gain in enumerate(ideal[:k], start=1):
            idcg += gain / np.log2(i + 1)

        if idcg > 0:
            scores.append(float(dcg / idcg))

    return float(np.mean(scores)) if scores else float("nan")


def event_precision_at_k(df: pd.DataFrame, score_col: str, target_col: str, k: int) -> float:
    vals: list[float] = []
    for _, group in df.groupby(EVENT_KEY):
        topk = group.nlargest(k, score_col)
        vals.append(float(topk[target_col].mean()))
    return float(np.mean(vals)) if vals else float("nan")


def event_overlap_at_k(df: pd.DataFrame, score_col: str, target_col: str, k: int) -> float:
    vals: list[float] = []
    for _, group in df.groupby(EVENT_KEY):
        pred_topk = set(group.nlargest(k, score_col)["driver"].tolist())
        actual_pos = set(group.loc[group[target_col] == 1, "driver"].tolist())
        denom = max(len(actual_pos), 1)
        vals.append(float(len(pred_topk & actual_pos) / denom))
    return float(np.mean(vals)) if vals else float("nan")


def event_dnf_probability_mae(df: pd.DataFrame, target_col: str, probability_col: str) -> float:
    event = (
        df.groupby(EVENT_KEY, as_index=False)
        .agg(actual_dnf=(target_col, "sum"), predicted_dnf=(probability_col, "sum"))
    )
    return float(np.mean(np.abs(event["actual_dnf"] - event["predicted_dnf"])))


def event_dnf_count_mae(df: pd.DataFrame, target_col: str, prediction_col: str) -> float:
    event = (
        df.groupby(EVENT_KEY, as_index=False)
        .agg(actual_dnf=(target_col, "sum"), predicted_dnf=(prediction_col, "sum"))
    )
    return float(np.mean(np.abs(event["actual_dnf"] - event["predicted_dnf"])))


def event_label_topk_hit_rate(df: pd.DataFrame, score_col: str, label_name: str, k: int) -> float:
    hits: list[float] = []
    for _, group in df.groupby(EVENT_KEY):
        pred_topk = set(group.nlargest(k, score_col)["driver"].tolist())
        truth = set(group.loc[group["actual_outcome"] == label_name, "driver"].tolist())
        if not truth:
            continue
        hits.append(float(len(pred_topk & truth) / len(truth)))
    return float(np.mean(hits)) if hits else float("nan")


def binary_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_score: pd.Series | None = None,
) -> dict[str, float | None]:
    y_true_arr = y_true.astype(int).to_numpy()
    y_pred_arr = y_pred.astype(int).to_numpy()

    metrics: dict[str, float | None] = {
        "accuracy": safe_float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": safe_float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": safe_float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1": safe_float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
    }

    if y_score is None:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None
        return metrics

    y_score_arr = pd.to_numeric(y_score, errors="coerce").fillna(0.0).to_numpy()
    if len(np.unique(y_true_arr)) < 2:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None
    else:
        metrics["roc_auc"] = safe_float(roc_auc_score(y_true_arr, y_score_arr))
        metrics["average_precision"] = safe_float(average_precision_score(y_true_arr, y_score_arr))
    return metrics


def multiclass_metrics(
    actual: pd.Series,
    predicted: pd.Series,
    score_for_topk: pd.Series,
    rows: pd.DataFrame,
) -> dict[str, Any]:
    y_true = actual.astype(str)
    y_pred = predicted.astype(str)

    dnf_true = (y_true == "DNF").astype(int)
    dnf_pred = (y_pred == "DNF").astype(int)

    eval_df = rows.copy()
    eval_df["actual_outcome"] = y_true
    eval_df["predicted_outcome"] = y_pred
    eval_df["ranking_score"] = pd.to_numeric(score_for_topk, errors="coerce").fillna(0.0)

    return {
        "accuracy": safe_float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": safe_float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": safe_float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": safe_float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "dnf_precision": safe_float(precision_score(dnf_true, dnf_pred, zero_division=0)),
        "dnf_recall": safe_float(recall_score(dnf_true, dnf_pred, zero_division=0)),
        "win_top1_hit_rate": safe_float(event_label_topk_hit_rate(eval_df, "ranking_score", "WIN", 1)),
        "podium_top3_hit_rate": safe_float(
            event_label_topk_hit_rate(eval_df, "ranking_score", "PODIUM", 3)
        ),
    }


def build_dnf_risk_score(df: pd.DataFrame, fallback: float) -> pd.Series:
    weighted_cols = [
        ("driver_dnf_rate_last5", 0.45),
        ("team_dnf_rate_last5", 0.25),
        ("driver_event_dnf_rate_hist", 0.20),
        ("team_event_dnf_rate_hist", 0.10),
    ]

    total_weight = 0.0
    score = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    for col, weight in weighted_cols:
        if col not in df.columns:
            continue
        score += weight * to_numeric(df[col], fallback)
        total_weight += weight

    if total_weight <= 0:
        return pd.Series(np.full(len(df), fallback, dtype=float), index=df.index)

    return (score / total_weight).clip(lower=0.0, upper=1.0)


def build_points_prior_score(df: pd.DataFrame, fallback_rate: float) -> pd.Series:
    driver_rate = to_numeric(df.get("driver_points_rate_last5", pd.Series(index=df.index, dtype=float)), fallback_rate)
    team_rate = to_numeric(df.get("team_points_rate_last5", pd.Series(index=df.index, dtype=float)), fallback_rate)
    driver_event = to_numeric(
        df.get("driver_event_points_rate_hist", pd.Series(index=df.index, dtype=float)), fallback_rate
    )
    team_event = to_numeric(
        df.get("team_event_points_rate_hist", pd.Series(index=df.index, dtype=float)), fallback_rate
    )

    grid_raw = pd.to_numeric(df.get("grid_position_target", pd.Series(index=df.index, dtype=float)), errors="coerce")
    event_grid_max = (
        grid_raw.groupby([df["year"], df["round"], df["event_name"]]).transform("max").replace(0, np.nan)
    )
    grid_strength = 1.0 - ((grid_raw - 1.0) / (event_grid_max - 1.0))
    grid_strength = grid_strength.fillna(0.5).clip(0.0, 1.0)

    return (
        0.40 * driver_rate
        + 0.25 * team_rate
        + 0.20 * driver_event
        + 0.05 * team_event
        + 0.10 * grid_strength
    ).clip(0.0, 1.0)


def predict_topk_by_event(df: pd.DataFrame, score_col: str, k: int, out_col: str) -> pd.DataFrame:
    pred = df.copy()
    pred["event_rank"] = pred.groupby(EVENT_KEY)[score_col].rank(method="first", ascending=False)
    pred[out_col] = (pred["event_rank"] <= int(k)).astype(int)
    return pred


def evaluate_grid_predictions(df: pd.DataFrame, score_col: str, position_col: str) -> dict[str, float | None]:
    winners = df.loc[df["actual_grid_position"] == 1]
    winner_accuracy = (
        float((winners[position_col] == 1).mean()) if len(winners) else float("nan")
    )
    return {
        "rows": int(len(df)),
        "mae_ranked_position": safe_float(
            mean_absolute_error(df["actual_grid_position"], df[position_col])
        ),
        "winner_accuracy": safe_float(winner_accuracy),
        "mean_event_spearman": safe_float(
            mean_event_spearman(df, "actual_grid_position", position_col)
        ),
        "ndcg_at_3": safe_float(ndcg_at_k(df, score_col, "actual_grid_position", k=3)),
        "ndcg_at_10": safe_float(ndcg_at_k(df, score_col, "actual_grid_position", k=10)),
    }


def evaluate_points_predictions(
    df: pd.DataFrame,
    pred_col: str,
    score_col: str,
) -> dict[str, float | None]:
    metrics = binary_metrics(df["points_target"], df[pred_col], df[score_col])
    metrics.update(
        {
            "rows": int(len(df)),
            "event_precision_at_10": safe_float(
                event_precision_at_k(df, score_col, "points_target", k=10)
            ),
            "event_overlap_at_10": safe_float(
                event_overlap_at_k(df, score_col, "points_target", k=10)
            ),
        }
    )
    return metrics


def evaluate_dnf_predictions(
    df: pd.DataFrame,
    pred_col: str,
    score_col: str,
) -> dict[str, float | None]:
    metrics = binary_metrics(df["dnf_target"], df[pred_col], df[score_col])
    metrics.update(
        {
            "rows": int(len(df)),
            "test_dnf_rate": safe_float(float(df["dnf_target"].mean())),
            "predicted_dnf_rate": safe_float(float(df[pred_col].mean())),
            "event_dnf_mae": safe_float(event_dnf_probability_mae(df, "dnf_target", score_col)),
            "event_dnf_count_mae": safe_float(event_dnf_count_mae(df, "dnf_target", pred_col)),
        }
    )
    return metrics


def grid_rule_labels(grid_positions: pd.Series) -> np.ndarray:
    grid = pd.to_numeric(grid_positions, errors="coerce").fillna(20).to_numpy(dtype=float)
    return np.where(
        grid <= 1,
        "WIN",
        np.where(grid <= 3, "PODIUM", np.where(grid <= 10, "POINTS", "NO_POINTS")),
    )


def select_best_baseline(
    baselines: dict[str, dict[str, Any]],
    primary_metric: str,
    higher_is_better: bool,
) -> tuple[str | None, dict[str, Any] | None]:
    valid: list[tuple[str, dict[str, Any], float]] = []
    for name, metrics in baselines.items():
        value = metrics.get(primary_metric)
        if value is None:
            continue
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            continue
        valid.append((name, metrics, numeric))

    if not valid:
        return None, None

    if higher_is_better:
        best = max(valid, key=lambda item: item[2])
    else:
        best = min(valid, key=lambda item: item[2])

    return best[0], best[1]


def gain(model_value: float | None, baseline_value: float | None, higher_is_better: bool) -> float | None:
    if model_value is None or baseline_value is None:
        return None
    if higher_is_better:
        return safe_float(model_value - baseline_value)
    return safe_float(baseline_value - model_value)


def model_vs_baselines_summary(
    model_metrics: dict[str, Any],
    baseline_metrics: dict[str, dict[str, Any]],
    primary_metric: str,
    higher_is_better: bool,
    delta_metrics: dict[str, bool],
) -> dict[str, Any]:
    best_name, best_metrics = select_best_baseline(
        baseline_metrics, primary_metric=primary_metric, higher_is_better=higher_is_better
    )

    deltas: dict[str, float | None] = {}
    if best_metrics is not None:
        for metric_name, metric_higher_is_better in delta_metrics.items():
            deltas[metric_name] = gain(
                model_metrics.get(metric_name),
                best_metrics.get(metric_name),
                higher_is_better=metric_higher_is_better,
            )

    return {
        "primary_metric": primary_metric,
        "primary_higher_is_better": higher_is_better,
        "best_baseline": best_name,
        "model": model_metrics,
        "baselines": baseline_metrics,
        "model_gain_vs_best_baseline": deltas,
    }


def clean_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [clean_for_json(v) for v in obj]
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        cleaned = safe_float(float(obj))
        return cleaned
    return obj


def benchmark_grid(args: argparse.Namespace) -> dict[str, Any]:
    model_pred = pd.read_csv(args.grid_predictions)
    model_eval = model_pred[["year", "round", "event_name", "driver", "actual_grid_position", "predicted_grid_score", "predicted_grid_position"]].copy()
    model_eval["actual_grid_position"] = to_numeric(model_eval["actual_grid_position"], 20)
    model_eval["predicted_grid_position"] = to_numeric(model_eval["predicted_grid_position"], 20).astype(int)
    model_eval["predicted_grid_score"] = to_numeric(model_eval["predicted_grid_score"], 0.0)

    grid_data = build_driver_event_dataset(args.data_root)
    train_mask = (grid_data["year"] >= args.train_start_year) & (grid_data["year"] <= args.train_end_year)
    test_mask = grid_data["year"] == args.test_year

    train_df = grid_data.loc[train_mask].copy()
    test_df = grid_data.loc[test_mask].copy()

    if test_df.empty:
        raise RuntimeError("No test rows found for grid baseline benchmark.")

    driver_prior = train_df.groupby("driver")["grid_position_target"].mean()
    team_prior = train_df.groupby("team")["grid_position_target"].mean()
    event_prior = train_df.groupby("event_name")["grid_position_target"].mean()
    global_prior = float(train_df["grid_position_target"].mean())

    prior_df = test_df[["year", "round", "event_name", "driver", "team", "grid_position_target"]].copy()
    prior_df["driver_prior"] = prior_df["driver"].map(driver_prior)
    prior_df["team_prior"] = prior_df["team"].map(team_prior)
    prior_df["event_prior"] = prior_df["event_name"].map(event_prior)
    prior_df["driver_prior"] = prior_df["driver_prior"].fillna(global_prior)
    prior_df["team_prior"] = prior_df["team_prior"].fillna(global_prior)
    prior_df["event_prior"] = prior_df["event_prior"].fillna(global_prior)

    prior_df["baseline_score"] = -(
        0.60 * prior_df["driver_prior"]
        + 0.30 * prior_df["team_prior"]
        + 0.10 * prior_df["event_prior"]
    )
    prior_df = add_event_rank(prior_df, "baseline_score", "predicted_grid_position")
    prior_df = prior_df.rename(columns={"grid_position_target": "actual_grid_position"})

    pace_cols = [
        col
        for col in ["fp3_best_lap_seconds", "fp2_best_lap_seconds", "fp1_best_lap_seconds"]
        if col in test_df.columns
    ]
    if not pace_cols:
        raise RuntimeError("No FP pace columns available for pace baseline benchmark.")

    pace_df = test_df[["year", "round", "event_name", "driver", "grid_position_target"] + pace_cols].copy()
    pace_values = pace_df[pace_cols].apply(pd.to_numeric, errors="coerce")
    pace_df["pace_seconds"] = pace_values.bfill(axis=1).iloc[:, 0]
    event_median = pace_df.groupby(EVENT_KEY)["pace_seconds"].transform("median")
    global_median = float(np.nanmedian(pace_df["pace_seconds"].to_numpy(dtype=float)))
    pace_df["pace_seconds"] = pace_df["pace_seconds"].fillna(event_median).fillna(global_median)
    pace_df["baseline_score"] = -pace_df["pace_seconds"]
    pace_df = add_event_rank(pace_df, "baseline_score", "predicted_grid_position")
    pace_df = pace_df.rename(columns={"grid_position_target": "actual_grid_position"})

    model_metrics = evaluate_grid_predictions(model_eval, "predicted_grid_score", "predicted_grid_position")

    prior_eval = model_eval[ROW_KEY + ["actual_grid_position"]].merge(
        prior_df[ROW_KEY + ["baseline_score", "predicted_grid_position"]],
        on=ROW_KEY,
        how="inner",
    )
    pace_eval = model_eval[ROW_KEY + ["actual_grid_position"]].merge(
        pace_df[ROW_KEY + ["baseline_score", "predicted_grid_position"]],
        on=ROW_KEY,
        how="inner",
    )

    baseline_metrics = {
        "driver_team_prior": evaluate_grid_predictions(prior_eval, "baseline_score", "predicted_grid_position"),
        "fp_pace_rank": evaluate_grid_predictions(pace_eval, "baseline_score", "predicted_grid_position"),
    }

    return model_vs_baselines_summary(
        model_metrics=model_metrics,
        baseline_metrics=baseline_metrics,
        primary_metric="mean_event_spearman",
        higher_is_better=True,
        delta_metrics={
            "mean_event_spearman": True,
            "ndcg_at_10": True,
            "mae_ranked_position": False,
            "winner_accuracy": True,
        },
    )


def benchmark_points(args: argparse.Namespace) -> dict[str, Any]:
    model_pred = pd.read_csv(args.points_predictions)
    model_eval = model_pred[
        ["year", "round", "event_name", "driver", "points_target", "points_probability", "predicted_points_target"]
    ].copy()
    model_eval["points_target"] = to_numeric(model_eval["points_target"], 0).astype(int)
    model_eval["points_probability"] = to_numeric(model_eval["points_probability"], 0.0)
    model_eval["predicted_points_target"] = to_numeric(model_eval["predicted_points_target"], 0).astype(int)

    dataset = build_points_dataset(args.data_root)
    train_mask = (dataset["year"] >= args.train_start_year) & (dataset["year"] <= args.train_end_year)
    test_mask = dataset["year"] == args.test_year

    train_df = dataset.loc[train_mask].copy()
    test_df = dataset.loc[test_mask].copy()

    if test_df.empty:
        raise RuntimeError("No test rows found for points baseline benchmark.")

    model_metrics = evaluate_points_predictions(
        model_eval,
        pred_col="predicted_points_target",
        score_col="points_probability",
    )

    baseline_grid = test_df[["year", "round", "event_name", "driver", "points_target", "grid_position_target"]].copy()
    baseline_grid["baseline_score"] = -to_numeric(baseline_grid["grid_position_target"], 20)
    baseline_grid = predict_topk_by_event(baseline_grid, "baseline_score", k=10, out_col="predicted_points_target")

    avg_points_rate = float(train_df["points_target"].mean()) if not train_df.empty else 0.5
    baseline_prior = test_df[
        [
            "year",
            "round",
            "event_name",
            "driver",
            "points_target",
            "grid_position_target",
            "driver_points_rate_last5",
            "team_points_rate_last5",
            "driver_event_points_rate_hist",
            "team_event_points_rate_hist",
        ]
    ].copy()
    baseline_prior["baseline_score"] = build_points_prior_score(baseline_prior, fallback_rate=avg_points_rate)
    baseline_prior = predict_topk_by_event(
        baseline_prior, "baseline_score", k=10, out_col="predicted_points_target"
    )

    base_grid_eval = model_eval[ROW_KEY + ["points_target"]].merge(
        baseline_grid[ROW_KEY + ["baseline_score", "predicted_points_target"]],
        on=ROW_KEY,
        how="inner",
    )
    base_prior_eval = model_eval[ROW_KEY + ["points_target"]].merge(
        baseline_prior[ROW_KEY + ["baseline_score", "predicted_points_target"]],
        on=ROW_KEY,
        how="inner",
    )

    baseline_metrics = {
        "grid_top10": evaluate_points_predictions(
            base_grid_eval,
            pred_col="predicted_points_target",
            score_col="baseline_score",
        ),
        "history_prior_top10": evaluate_points_predictions(
            base_prior_eval,
            pred_col="predicted_points_target",
            score_col="baseline_score",
        ),
    }

    return model_vs_baselines_summary(
        model_metrics=model_metrics,
        baseline_metrics=baseline_metrics,
        primary_metric="event_overlap_at_10",
        higher_is_better=True,
        delta_metrics={
            "event_overlap_at_10": True,
            "event_precision_at_10": True,
            "f1": True,
            "roc_auc": True,
        },
    )


def benchmark_dnf(args: argparse.Namespace) -> dict[str, Any]:
    model_pred = pd.read_csv(args.dnf_predictions)
    model_eval = model_pred[
        ["year", "round", "event_name", "driver", "dnf_target", "dnf_probability", "predicted_dnf_target"]
    ].copy()
    model_eval["dnf_target"] = to_numeric(model_eval["dnf_target"], 0).astype(int)
    model_eval["dnf_probability"] = to_numeric(model_eval["dnf_probability"], 0.0)
    model_eval["predicted_dnf_target"] = to_numeric(model_eval["predicted_dnf_target"], 0).astype(int)

    dataset = build_dnf_dataset(args.data_root)
    train_mask = (dataset["year"] >= args.train_start_year) & (dataset["year"] <= args.train_end_year)
    test_mask = dataset["year"] == args.test_year

    train_df = dataset.loc[train_mask].copy()
    test_df = dataset.loc[test_mask].copy()

    if test_df.empty:
        raise RuntimeError("No test rows found for DNF baseline benchmark.")

    model_metrics = evaluate_dnf_predictions(
        model_eval,
        pred_col="predicted_dnf_target",
        score_col="dnf_probability",
    )

    baseline_finish = model_eval[["year", "round", "event_name", "driver", "dnf_target"]].copy()
    baseline_finish["baseline_score"] = 0.0
    baseline_finish["predicted_dnf_target"] = 0

    avg_dnf_rate = float(train_df["dnf_target"].mean()) if not train_df.empty else 0.15
    avg_dnf_count = (
        int(round(train_df.groupby(EVENT_KEY)["dnf_target"].sum().mean())) if not train_df.empty else 2
    )
    avg_dnf_count = max(avg_dnf_count, 1)

    baseline_risk = test_df[
        [
            "year",
            "round",
            "event_name",
            "driver",
            "dnf_target",
            "driver_dnf_rate_last5",
            "team_dnf_rate_last5",
            "driver_event_dnf_rate_hist",
            "team_event_dnf_rate_hist",
        ]
    ].copy()
    baseline_risk["baseline_score"] = build_dnf_risk_score(baseline_risk, fallback=avg_dnf_rate)
    baseline_risk = predict_topk_by_event(
        baseline_risk,
        "baseline_score",
        k=avg_dnf_count,
        out_col="predicted_dnf_target",
    )

    finish_eval = model_eval[ROW_KEY + ["dnf_target"]].merge(
        baseline_finish[ROW_KEY + ["baseline_score", "predicted_dnf_target"]],
        on=ROW_KEY,
        how="inner",
    )
    risk_eval = model_eval[ROW_KEY + ["dnf_target"]].merge(
        baseline_risk[ROW_KEY + ["baseline_score", "predicted_dnf_target"]],
        on=ROW_KEY,
        how="inner",
    )

    baseline_metrics = {
        "always_finish": evaluate_dnf_predictions(
            finish_eval,
            pred_col="predicted_dnf_target",
            score_col="baseline_score",
        ),
        "history_risk_topk": evaluate_dnf_predictions(
            risk_eval,
            pred_col="predicted_dnf_target",
            score_col="baseline_score",
        ),
    }

    return model_vs_baselines_summary(
        model_metrics=model_metrics,
        baseline_metrics=baseline_metrics,
        primary_metric="f1",
        higher_is_better=True,
        delta_metrics={
            "f1": True,
            "recall": True,
            "precision": True,
            "event_dnf_count_mae": False,
            "roc_auc": True,
        },
    )


def benchmark_outcome(args: argparse.Namespace) -> dict[str, Any]:
    model_pred = pd.read_csv(args.outcome_predictions)
    model_eval = model_pred[
        [
            "year",
            "round",
            "event_name",
            "driver",
            "actual_outcome",
            "predicted_outcome",
            "win_probability",
        ]
    ].copy()

    dataset = build_points_dataset(args.data_root)
    dataset = dataset.copy()
    dataset["actual_outcome"] = make_outcome_class(dataset)

    train_mask = (dataset["year"] >= args.train_start_year) & (dataset["year"] <= args.train_end_year)
    test_mask = dataset["year"] == args.test_year

    train_df = dataset.loc[train_mask].copy()
    test_df = dataset.loc[test_mask].copy()

    if test_df.empty:
        raise RuntimeError("No test rows found for multiclass outcome baseline benchmark.")

    model_metrics = multiclass_metrics(
        actual=model_eval["actual_outcome"],
        predicted=model_eval["predicted_outcome"],
        score_for_topk=model_eval["win_probability"],
        rows=model_eval[["year", "round", "event_name", "driver"]],
    )
    model_metrics["rows"] = int(len(model_eval))

    avg_dnf_rate = float(train_df["dnf_target"].mean()) if not train_df.empty else 0.15
    avg_dnf_count = (
        int(round(train_df.groupby(EVENT_KEY)["dnf_target"].sum().mean())) if not train_df.empty else 2
    )
    avg_dnf_count = max(avg_dnf_count, 1)

    baseline_grid = test_df[["year", "round", "event_name", "driver", "actual_outcome", "grid_position_target"]].copy()
    baseline_grid["predicted_outcome"] = grid_rule_labels(baseline_grid["grid_position_target"])
    baseline_grid["ranking_score"] = -to_numeric(baseline_grid["grid_position_target"], 20)

    baseline_risk = test_df[
        [
            "year",
            "round",
            "event_name",
            "driver",
            "actual_outcome",
            "grid_position_target",
            "driver_dnf_rate_last5",
            "team_dnf_rate_last5",
            "driver_event_dnf_rate_hist",
            "team_event_dnf_rate_hist",
        ]
    ].copy()
    baseline_risk["predicted_outcome"] = grid_rule_labels(baseline_risk["grid_position_target"])
    baseline_risk["dnf_risk_score"] = build_dnf_risk_score(baseline_risk, fallback=avg_dnf_rate)
    baseline_risk = add_event_rank(baseline_risk, "dnf_risk_score", "dnf_risk_rank")
    baseline_risk.loc[baseline_risk["dnf_risk_rank"] <= avg_dnf_count, "predicted_outcome"] = "DNF"
    baseline_risk["ranking_score"] = -to_numeric(baseline_risk["grid_position_target"], 20)

    base_grid_eval = model_eval[ROW_KEY + ["actual_outcome"]].merge(
        baseline_grid[ROW_KEY + ["predicted_outcome", "ranking_score"]],
        on=ROW_KEY,
        how="inner",
    )
    base_risk_eval = model_eval[ROW_KEY + ["actual_outcome"]].merge(
        baseline_risk[ROW_KEY + ["predicted_outcome", "ranking_score"]],
        on=ROW_KEY,
        how="inner",
    )

    baseline_metrics = {
        "grid_rule_no_dnf": multiclass_metrics(
            actual=base_grid_eval["actual_outcome"],
            predicted=base_grid_eval["predicted_outcome"],
            score_for_topk=base_grid_eval["ranking_score"],
            rows=base_grid_eval[ROW_KEY],
        ),
        "grid_rule_plus_dnf_risk": multiclass_metrics(
            actual=base_risk_eval["actual_outcome"],
            predicted=base_risk_eval["predicted_outcome"],
            score_for_topk=base_risk_eval["ranking_score"],
            rows=base_risk_eval[ROW_KEY],
        ),
    }
    for metrics in baseline_metrics.values():
        metrics["rows"] = int(len(base_grid_eval))

    return model_vs_baselines_summary(
        model_metrics=model_metrics,
        baseline_metrics=baseline_metrics,
        primary_metric="macro_f1",
        higher_is_better=True,
        delta_metrics={
            "macro_f1": True,
            "accuracy": True,
            "dnf_recall": True,
            "dnf_precision": True,
            "win_top1_hit_rate": True,
        },
    )


def main() -> None:
    args = parse_args()

    results = {
        "config": {
            "train_start_year": args.train_start_year,
            "train_end_year": args.train_end_year,
            "test_year": args.test_year,
            "grid_predictions": str(args.grid_predictions),
            "points_predictions": str(args.points_predictions),
            "dnf_predictions": str(args.dnf_predictions),
            "outcome_predictions": str(args.outcome_predictions),
        },
        "grid_ranking": benchmark_grid(args),
        "race_points": benchmark_points(args),
        "dnf": benchmark_dnf(args),
        "race_outcome_multiclass": benchmark_outcome(args),
    }

    payload = clean_for_json(results)
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Saved benchmark summary to: {args.out_file}")
    for section in ["grid_ranking", "race_points", "dnf", "race_outcome_multiclass"]:
        section_data = payload[section]
        print(f"\n[{section}]")
        print(f"Primary metric: {section_data['primary_metric']}")
        print(f"Best baseline: {section_data['best_baseline']}")
        gain_map = section_data.get("model_gain_vs_best_baseline", {})
        if gain_map:
            for metric, value in gain_map.items():
                print(f"- model gain on {metric}: {value}")


if __name__ == "__main__":
    main()
