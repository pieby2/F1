from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

STREET_EVENT_KEYWORDS = {
    "Monaco",
    "Singapore",
    "Las Vegas",
    "Azerbaijan",
    "Saudi Arabian",
    "Miami",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run focused error analysis on grid-position model predictions."
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=Path("data") / "model_outputs" / "grid_position_predictions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "model_outputs" / "grid_error_analysis",
    )
    return parser.parse_args()


def infer_track_type(event_name: str) -> str:
    for keyword in STREET_EVENT_KEYWORDS:
        if keyword.lower() in event_name.lower():
            return "street"
    return "permanent_or_other"


def top3_overlap(group: pd.DataFrame) -> float:
    actual_top3 = set(group.nsmallest(3, "actual_grid_position")["driver"].tolist())
    predicted_top3 = set(group.nsmallest(3, "predicted_grid_position")["driver"].tolist())
    return float(len(actual_top3 & predicted_top3) / 3.0)


def winner_hit(group: pd.DataFrame) -> int:
    actual_winner = group.loc[group["actual_grid_position"] == 1, "driver"]
    pred_winner = group.loc[group["predicted_grid_position"] == 1, "driver"]
    if actual_winner.empty or pred_winner.empty:
        return 0
    return int(actual_winner.iloc[0] == pred_winner.iloc[0])


def build_event_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    event_cols = ["year", "round", "event_name"]
    rows: list[dict[str, Any]] = []

    for event_key, group in preds.groupby(event_cols):
        year, round_no, event_name = event_key
        corr = group["actual_grid_position"].corr(group["predicted_grid_position"], method="spearman")
        rows.append(
            {
                "year": int(year),
                "round": int(round_no),
                "event_name": str(event_name),
                "track_type": infer_track_type(str(event_name)),
                "rows": int(len(group)),
                "mae_rank": float(group["abs_rank_error"].mean()),
                "winner_hit": winner_hit(group),
                "top3_overlap": top3_overlap(group),
                "spearman": float(corr) if pd.notna(corr) else float("nan"),
            }
        )

    return pd.DataFrame(rows).sort_values(["year", "round"]).reset_index(drop=True)


def safe_top3_recall(group: pd.DataFrame) -> float:
    actual = group["actual_top3"]
    if actual.sum() == 0:
        return float("nan")
    return float((group["actual_top3"] & group["predicted_top3"]).sum() / actual.sum())


def build_driver_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for driver, group in preds.groupby("driver"):
        rows.append(
            {
                "driver": str(driver),
                "rows": int(len(group)),
                "mae_rank": float(group["abs_rank_error"].mean()),
                "mean_rank_bias": float(group["rank_error"].mean()),
                "top3_recall": safe_top3_recall(group),
                "average_actual_grid": float(group["actual_grid_position"].mean()),
                "average_predicted_grid": float(group["predicted_grid_position"].mean()),
            }
        )

    return pd.DataFrame(rows).sort_values("mae_rank", ascending=False).reset_index(drop=True)


def build_team_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for team, group in preds.groupby("team"):
        rows.append(
            {
                "team": str(team),
                "rows": int(len(group)),
                "mae_rank": float(group["abs_rank_error"].mean()),
                "mean_rank_bias": float(group["rank_error"].mean()),
                "top3_recall": safe_top3_recall(group),
                "average_actual_grid": float(group["actual_grid_position"].mean()),
                "average_predicted_grid": float(group["predicted_grid_position"].mean()),
            }
        )

    return pd.DataFrame(rows).sort_values("mae_rank", ascending=False).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    preds = pd.read_csv(args.predictions_path)

    required_cols = {
        "year",
        "round",
        "event_name",
        "driver",
        "team",
        "actual_grid_position",
        "predicted_grid_position",
    }
    missing = required_cols - set(preds.columns)
    if missing:
        raise RuntimeError(f"Predictions file is missing required columns: {sorted(missing)}")

    preds["rank_error"] = preds["predicted_grid_position"] - preds["actual_grid_position"]
    preds["abs_rank_error"] = preds["rank_error"].abs()
    preds["actual_top3"] = preds["actual_grid_position"] <= 3
    preds["predicted_top3"] = preds["predicted_grid_position"] <= 3

    event_metrics = build_event_metrics(preds)
    driver_metrics = build_driver_metrics(preds)
    team_metrics = build_team_metrics(preds)

    segment_summary = (
        event_metrics.groupby("track_type", dropna=False)
        .agg(
            events=("event_name", "count"),
            mean_mae=("mae_rank", "mean"),
            mean_spearman=("spearman", "mean"),
            winner_hit_rate=("winner_hit", "mean"),
            mean_top3_overlap=("top3_overlap", "mean"),
        )
        .reset_index()
    )

    overall_summary = {
        "rows": int(len(preds)),
        "events": int(event_metrics["event_name"].count()),
        "overall_mae_rank": float(preds["abs_rank_error"].mean()),
        "overall_mean_spearman": float(event_metrics["spearman"].mean()),
        "overall_winner_hit_rate": float(event_metrics["winner_hit"].mean()),
        "overall_top3_overlap": float(event_metrics["top3_overlap"].mean()),
        "worst_events_by_mae": event_metrics.nlargest(5, "mae_rank")[
            ["year", "round", "event_name", "mae_rank", "spearman", "winner_hit"]
        ].to_dict("records"),
        "most_overestimated_drivers": driver_metrics.nlargest(5, "mean_rank_bias")[
            ["driver", "mean_rank_bias", "mae_rank", "rows"]
        ].to_dict("records"),
        "most_underestimated_drivers": driver_metrics.nsmallest(5, "mean_rank_bias")[
            ["driver", "mean_rank_bias", "mae_rank", "rows"]
        ].to_dict("records"),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    event_metrics.to_csv(args.output_dir / "event_metrics.csv", index=False)
    driver_metrics.to_csv(args.output_dir / "driver_metrics.csv", index=False)
    team_metrics.to_csv(args.output_dir / "team_metrics.csv", index=False)
    segment_summary.to_csv(args.output_dir / "segment_summary.csv", index=False)

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(overall_summary, handle, indent=2)

    print("Error analysis complete.")
    print(f"Output directory: {args.output_dir}")
    print(f"Overall MAE rank: {overall_summary['overall_mae_rank']:.3f}")
    print(f"Overall mean Spearman: {overall_summary['overall_mean_spearman']:.3f}")
    print(f"Winner hit rate: {overall_summary['overall_winner_hit_rate']:.3f}")
    print(f"Top-3 overlap: {overall_summary['overall_top3_overlap']:.3f}")


if __name__ == "__main__":
    main()
