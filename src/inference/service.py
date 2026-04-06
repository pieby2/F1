from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fastf1
import joblib
import numpy as np
import pandas as pd

from train_grid_position_model import build_driver_event_dataset
from train_race_points_model import build_model_dataset as build_points_dataset

OUTCOME_CLASSES = ["WIN", "PODIUM", "POINTS", "NO_POINTS", "DNF"]


class InferenceService:
    """Reusable model-serving service for post-qualifying race predictions."""

    def __init__(
        self,
        data_root: Path | str = Path("data") / "fastf1_csv",
        models_root: Path | str = Path("models"),
        cache_dir: Path | str | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.models_root = Path(models_root)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else self.data_root / "_api_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.points_model = self._load_model("race_points_classifier.joblib")
        self.dnf_model = self._load_model("dnf_classifier.joblib")
        self.outcome_model = self._load_model("race_outcome_multiclass.joblib")

        self.grid_rank_bundle = self._load_optional_model("grid_position_ranker.joblib")

        self.grid_dataset = self._load_grid_dataset()
        self.points_dataset = self._load_points_dataset()
        self.driver_name_map = self._build_driver_name_map()
        self.schedule_cache: dict[int, dict[int, dict[str, str | None]]] = {}

    @staticmethod
    def _truthy_env(name: str, default: str = "1") -> bool:
        value = os.getenv(name, default)
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _cache_is_usable(cache_stem: Path) -> bool:
        parquet_path = cache_stem.with_suffix(".parquet")
        pickle_path = cache_stem.with_suffix(".pkl")
        return (
            (parquet_path.exists() and parquet_path.stat().st_size > 0)
            or (pickle_path.exists() and pickle_path.stat().st_size > 0)
        )

    @staticmethod
    def _safe_read_cache(cache_stem: Path) -> pd.DataFrame | None:
        parquet_path = cache_stem.with_suffix(".parquet")
        pickle_path = cache_stem.with_suffix(".pkl")

        if parquet_path.exists() and parquet_path.stat().st_size > 0:
            try:
                return pd.read_parquet(parquet_path)
            except Exception:
                pass

        if pickle_path.exists() and pickle_path.stat().st_size > 0:
            try:
                return pd.read_pickle(pickle_path)
            except Exception:
                pass

        return None

    @staticmethod
    def _safe_write_cache(frame: pd.DataFrame, cache_stem: Path) -> None:
        parquet_path = cache_stem.with_suffix(".parquet")
        pickle_path = cache_stem.with_suffix(".pkl")

        try:
            frame.to_parquet(parquet_path, index=False)
            if pickle_path.exists():
                pickle_path.unlink(missing_ok=True)
            return
        except Exception:
            pass

        try:
            frame.to_pickle(pickle_path)
        except Exception:
            # Cache write failures must not block inference.
            pass

    def _load_model(self, filename: str) -> Any:
        path = self.models_root / filename
        if not path.exists():
            raise FileNotFoundError(f"Required model file not found: {path}")
        return joblib.load(path)

    def _load_optional_model(self, filename: str) -> Any | None:
        path = self.models_root / filename
        if not path.exists():
            return None
        return joblib.load(path)

    def _load_grid_dataset(self) -> pd.DataFrame:
        cache_enabled = self._truthy_env("INFERENCE_ENABLE_DATASET_CACHE", "1")
        cache_stem = self.cache_dir / "grid_dataset"

        dataset: pd.DataFrame | None = None
        if cache_enabled and self._cache_is_usable(cache_stem):
            dataset = self._safe_read_cache(cache_stem)

        if dataset is None:
            dataset = build_driver_event_dataset(self.data_root)
            if cache_enabled:
                self._safe_write_cache(dataset, cache_stem)

        dataset = dataset.sort_values(["year", "round", "event_name", "driver"]).reset_index(drop=True)
        dataset["year"] = pd.to_numeric(dataset["year"], errors="coerce").astype("Int64")
        dataset["round"] = pd.to_numeric(dataset["round"], errors="coerce").astype("Int64")
        return dataset

    def _load_points_dataset(self) -> pd.DataFrame:
        cache_enabled = self._truthy_env("INFERENCE_ENABLE_DATASET_CACHE", "1")
        cache_stem = self.cache_dir / "points_dataset"

        dataset: pd.DataFrame | None = None
        if cache_enabled and self._cache_is_usable(cache_stem):
            dataset = self._safe_read_cache(cache_stem)

        if dataset is None:
            dataset = build_points_dataset(self.data_root)
            if cache_enabled:
                self._safe_write_cache(dataset, cache_stem)

        dataset = dataset.sort_values(["year", "round", "event_name", "driver"]).reset_index(drop=True)
        dataset["year"] = pd.to_numeric(dataset["year"], errors="coerce").astype("Int64")
        dataset["round"] = pd.to_numeric(dataset["round"], errors="coerce").astype("Int64")
        return dataset

    def _build_driver_name_map(self) -> dict[str, str]:
        race_files = sorted((self.data_root / "race_results").rglob("*.csv"))
        if not race_files:
            return {}

        frames: list[pd.DataFrame] = []
        for race_file in race_files:
            frame = pd.read_csv(
                race_file,
                usecols=["year", "round", "driver_code", "driver"],
            )
            if frame.empty:
                continue
            frame = frame.dropna(subset=["driver_code", "driver"])
            frames.append(frame)

        if not frames:
            return {}

        merged = pd.concat(frames, ignore_index=True)
        merged["year"] = pd.to_numeric(merged["year"], errors="coerce")
        merged["round"] = pd.to_numeric(merged["round"], errors="coerce")
        merged = merged.sort_values(["year", "round"])
        merged = merged.drop_duplicates(subset=["driver_code"], keep="last")
        return {
            str(row.driver_code): str(row.driver)
            for row in merged.itertuples(index=False)
        }

    @staticmethod
    def _extract_feature_columns(model: Any) -> list[str]:
        if isinstance(model, dict) and "feature_cols" in model:
            return [str(col) for col in model["feature_cols"]]

        if not hasattr(model, "named_steps"):
            return []

        preprocessor = model.named_steps.get("preprocessor")
        if preprocessor is None:
            return []

        cols: list[str] = []
        transformers = getattr(preprocessor, "transformers_", None)
        if transformers is not None:
            for _, _, transformer_cols in transformers:
                if isinstance(transformer_cols, str) and transformer_cols in {"drop", "remainder"}:
                    continue
                if isinstance(transformer_cols, (list, tuple, np.ndarray, pd.Index)):
                    cols.extend([str(c) for c in list(transformer_cols)])
                else:
                    cols.append(str(transformer_cols))
        elif hasattr(preprocessor, "feature_names_in_"):
            cols.extend([str(c) for c in preprocessor.feature_names_in_])

        return list(dict.fromkeys(cols))

    @staticmethod
    def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        out = frame.copy()
        for col in columns:
            if col not in out.columns:
                out[col] = pd.NA
        return out[columns]

    @staticmethod
    def _binary_positive_index(model: Any) -> int:
        class_labels: list[Any] = []
        if hasattr(model, "classes_"):
            class_labels = list(getattr(model, "classes_"))
        elif hasattr(model, "named_steps") and "model" in model.named_steps:
            class_labels = list(getattr(model.named_steps["model"], "classes_", []))

        for candidate in (1, "1", True, "True"):
            if candidate in class_labels:
                return class_labels.index(candidate)

        return -1

    def _predict_binary_proba(self, model: Any, frame: pd.DataFrame) -> np.ndarray:
        feature_cols = self._extract_feature_columns(model)
        if not feature_cols:
            return np.zeros(len(frame), dtype=float)

        model_input = self._ensure_columns(frame, feature_cols)
        probabilities = model.predict_proba(model_input)

        if probabilities.ndim != 2 or probabilities.shape[1] == 0:
            return np.zeros(len(frame), dtype=float)

        idx = self._binary_positive_index(model)
        return probabilities[:, idx].astype(float)

    def _predict_outcome_probabilities(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        model = self.outcome_model
        feature_cols = self._extract_feature_columns(model)
        model_input = self._ensure_columns(frame, feature_cols)
        probabilities = model.predict_proba(model_input)

        class_labels: list[str] = []
        if hasattr(model, "classes_"):
            class_labels = [str(label) for label in list(model.classes_)]
        elif hasattr(model, "named_steps") and "model" in model.named_steps:
            class_labels = [str(label) for label in list(getattr(model.named_steps["model"], "classes_", []))]

        prob_map: dict[str, np.ndarray] = {}
        for idx, label in enumerate(class_labels):
            prob_map[label.upper()] = probabilities[:, idx].astype(float)

        for label in OUTCOME_CLASSES:
            prob_map.setdefault(label, np.zeros(len(frame), dtype=float))

        return prob_map

    def _predict_grid_score(self, frame: pd.DataFrame) -> np.ndarray:
        bundle = self.grid_rank_bundle
        if not isinstance(bundle, dict):
            return np.zeros(len(frame), dtype=float)

        required = [str(col) for col in bundle.get("feature_cols", [])]
        if not required:
            return np.zeros(len(frame), dtype=float)

        preprocessor = bundle.get("preprocessor")
        ranker = bundle.get("ranker")
        if preprocessor is None or ranker is None:
            return np.zeros(len(frame), dtype=float)

        model_input = self._ensure_columns(frame, required)
        encoded = preprocessor.transform(model_input)
        scores = ranker.predict(encoded)
        return np.asarray(scores, dtype=float)

    def _load_schedule(self, season: int) -> dict[int, dict[str, str | None]]:
        if season in self.schedule_cache:
            return self.schedule_cache[season]

        result: dict[int, dict[str, str | None]] = {}
        try:
            schedule = fastf1.get_event_schedule(season, include_testing=False)
            schedule = schedule[["RoundNumber", "EventName", "EventDate"]].copy()
            schedule["RoundNumber"] = pd.to_numeric(schedule["RoundNumber"], errors="coerce")
            for row in schedule.itertuples(index=False):
                if pd.isna(row.RoundNumber):
                    continue
                round_number = int(row.RoundNumber)
                date_value = pd.to_datetime(row.EventDate, errors="coerce")
                result[round_number] = {
                    "event_name": str(row.EventName),
                    "date": date_value.date().isoformat() if pd.notna(date_value) else None,
                }
        except Exception:
            result = {}

        self.schedule_cache[season] = result
        return result

    def get_available_seasons(self) -> list[int]:
        years = sorted(
            {
                int(v)
                for v in pd.concat([self.grid_dataset["year"], self.points_dataset["year"]], ignore_index=True)
                .dropna()
                .astype(int)
                .tolist()
            }
        )
        return years

    def get_events(self, season: int) -> list[dict[str, Any]]:
        season_grid = self.grid_dataset.loc[self.grid_dataset["year"] == season, ["round", "event_name"]].copy()
        season_grid = season_grid.drop_duplicates(subset=["round", "event_name"]).sort_values("round")

        schedule = self._load_schedule(season)

        events: list[dict[str, Any]] = []
        if season_grid.empty and schedule:
            for round_number, meta in sorted(schedule.items(), key=lambda item: item[0]):
                has_grid = bool(
                    ((self.grid_dataset["year"] == season) & (self.grid_dataset["round"] == round_number)).any()
                )
                has_context = bool(
                    ((self.points_dataset["year"] == season) & (self.points_dataset["round"] == round_number)).any()
                )
                events.append(
                    {
                        "round": int(round_number),
                        "event_name": meta.get("event_name") or f"Round {round_number}",
                        "date": meta.get("date"),
                        "available_for_prediction": bool(has_grid),
                        "has_race_context": bool(has_context),
                    }
                )
            return events

        for row in season_grid.itertuples(index=False):
            round_number = int(row.round)
            meta = schedule.get(round_number, {})
            has_context = bool(
                ((self.points_dataset["year"] == season) & (self.points_dataset["round"] == round_number)).any()
            )
            events.append(
                {
                    "round": round_number,
                    "event_name": str(row.event_name),
                    "date": meta.get("date"),
                    "available_for_prediction": True,
                    "has_race_context": bool(has_context),
                }
            )

        return events

    def get_next_event(self) -> dict[str, Any] | None:
        today = datetime.now(timezone.utc).date()
        candidate_years = [today.year, today.year + 1]

        for season in candidate_years:
            schedule = self._load_schedule(season)
            if not schedule:
                continue

            dated_events: list[tuple[int, str, str]] = []
            for round_number, meta in schedule.items():
                raw_date = meta.get("date")
                if not raw_date:
                    continue
                parsed = pd.to_datetime(raw_date, errors="coerce")
                if pd.isna(parsed):
                    continue
                dated_events.append((int(round_number), str(meta.get("event_name") or ""), parsed.date().isoformat()))

            if not dated_events:
                continue

            dated_events = sorted(dated_events, key=lambda item: item[2])
            future = [item for item in dated_events if item[2] >= today.isoformat()]
            chosen = future[0] if future else dated_events[-1]

            round_number, event_name, iso_date = chosen
            available = bool(
                ((self.grid_dataset["year"] == season) & (self.grid_dataset["round"] == round_number)).any()
            )
            has_context = bool(
                ((self.points_dataset["year"] == season) & (self.points_dataset["round"] == round_number)).any()
            )
            return {
                "season": int(season),
                "round": int(round_number),
                "event_name": event_name,
                "date": iso_date,
                "available_for_prediction": bool(available),
                "has_race_context": bool(has_context),
            }

        return None

    def _event_feature_frame(self, season: int, round_number: int) -> tuple[pd.DataFrame, str, str]:
        from_points = self.points_dataset.loc[
            (self.points_dataset["year"] == season) & (self.points_dataset["round"] == round_number)
        ].copy()
        if not from_points.empty:
            event_name = str(from_points.iloc[0]["event_name"])
            from_points = from_points.sort_values(by=["grid_position_target", "driver"], na_position="last")
            return from_points.reset_index(drop=True), event_name, "full_context"

        from_grid = self.grid_dataset.loc[
            (self.grid_dataset["year"] == season) & (self.grid_dataset["round"] == round_number)
        ].copy()
        if from_grid.empty:
            raise ValueError(f"No event data available for season={season}, round={round_number}.")

        event_name = str(from_grid.iloc[0]["event_name"])
        from_grid = from_grid.sort_values(by=["grid_position_target", "driver"], na_position="last")
        return from_grid.reset_index(drop=True), event_name, "grid_only"

    def predict_race(self, season: int, round_number: int) -> dict[str, Any]:
        frame, event_name, feature_source = self._event_feature_frame(season, round_number)

        if "team" not in frame.columns:
            frame["team"] = "UNKNOWN"
        frame["team"] = frame["team"].fillna("UNKNOWN")

        points_prob = self._predict_binary_proba(self.points_model, frame)
        dnf_model_prob = self._predict_binary_proba(self.dnf_model, frame)
        outcome_prob = self._predict_outcome_probabilities(frame)

        p_win = outcome_prob["WIN"]
        p_podium = np.clip(outcome_prob["WIN"] + outcome_prob["PODIUM"], 0.0, 1.0)
        p_points_outcome = np.clip(
            outcome_prob["WIN"] + outcome_prob["PODIUM"] + outcome_prob["POINTS"],
            0.0,
            1.0,
        )
        p_points = np.clip(0.55 * points_prob + 0.45 * p_points_outcome, 0.0, 1.0)
        p_dnf = np.clip(0.65 * dnf_model_prob + 0.35 * outcome_prob["DNF"], 0.0, 1.0)

        grid_score = self._predict_grid_score(frame)
        if len(grid_score):
            norm = np.nanstd(grid_score)
            norm = float(norm) if np.isfinite(norm) and norm > 0 else 1.0
            grid_signal = (grid_score - np.nanmean(grid_score)) / norm
        else:
            grid_signal = np.zeros(len(frame), dtype=float)

        finish_score = (
            2.2 * p_win
            + 1.35 * p_podium
            + 0.9 * p_points
            - 1.15 * p_dnf
            + 0.15 * grid_signal
        )

        out = frame[["driver", "driver_number", "team", "grid_position_target"]].copy()
        out = out.rename(columns={"driver": "driver_code", "grid_position_target": "grid_position"})
        out["driver_code"] = out["driver_code"].astype(str)
        out["driver_number"] = out["driver_number"].astype(str)
        out["driver_name"] = out["driver_code"].map(self.driver_name_map).fillna(out["driver_code"])

        out["p_win"] = p_win
        out["p_podium"] = p_podium
        out["p_points"] = p_points
        out["p_dnf"] = p_dnf
        out["finish_score"] = finish_score

        out = out.sort_values(by=["finish_score", "grid_position", "driver_code"], ascending=[False, True, True])
        out["predicted_finish"] = np.arange(1, len(out) + 1, dtype=int)

        schedule = self._load_schedule(season)
        event_date = schedule.get(round_number, {}).get("date")

        drivers = [
            {
                "driver_code": str(row.driver_code),
                "driver_name": str(row.driver_name),
                "team": str(row.team),
                "grid_position": int(row.grid_position) if pd.notna(row.grid_position) else None,
                "predicted_finish": int(row.predicted_finish),
                "p_win": float(row.p_win),
                "p_podium": float(row.p_podium),
                "p_points": float(row.p_points),
                "p_dnf": float(row.p_dnf),
            }
            for row in out.itertuples(index=False)
        ]

        return {
            "season": int(season),
            "round": int(round_number),
            "event_name": event_name,
            "event_date": event_date,
            "feature_source": feature_source,
            "drivers": drivers,
        }
