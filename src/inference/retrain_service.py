"""Walk-forward retrain-then-predict engine for F1 race predictions.

Before predicting race X, trains models on data strictly before race X,
then uses practice/qualifying features from race X for inference.

KEY OPTIMIZATION: Datasets are built ONCE and cached to disk as parquet.
Retraining only re-fits sklearn models (fast) - no CSV re-reading.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42
TREE_ESTIMATORS = 200  # Reduced for faster retraining


def _build_sklearn_pipeline(numeric_cols: list[str], categorical_cols: list[str], estimator) -> Pipeline:
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
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])


def _regression_ensemble() -> VotingRegressor:
    return VotingRegressor(
        estimators=[
            ("rf", RandomForestRegressor(n_estimators=TREE_ESTIMATORS, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)),
            ("et", ExtraTreesRegressor(n_estimators=TREE_ESTIMATORS, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)),
            ("gbr", GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE)),
        ]
    )


def _classification_ensemble() -> VotingClassifier:
    return VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=TREE_ESTIMATORS, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
            ("et", ExtraTreesClassifier(n_estimators=TREE_ESTIMATORS, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
            ("gbr", GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)),
        ],
        voting="soft",
    )


def _get_feature_cols(dataset: pd.DataFrame, exclude_cols: set[str]) -> tuple[list[str], list[str], list[str]]:
    """Return (feature_cols, numeric_cols, categorical_cols)."""
    categorical = ["driver", "driver_number", "team", "event_name"]
    categorical = [c for c in categorical if c in dataset.columns and c not in exclude_cols]

    numeric = [
        c for c in dataset.columns
        if c not in exclude_cols and c not in categorical
        and pd.api.types.is_numeric_dtype(dataset[c])
    ]

    feature_cols = numeric + categorical
    return feature_cols, numeric, categorical


class WalkForwardPredictor:
    """Train models on data up to race X-1, predict race X."""

    def __init__(self, data_root: Path | str = Path("data") / "fastf1_csv") -> None:
        self.data_root = Path(data_root)
        self._dataset_cache: pd.DataFrame | None = None
        self._dataset_hash: str | None = None
        self._model_cache: dict[str, dict[str, Any]] = {}

    def _cache_key(self, season: int, round_number: int) -> str:
        return f"{season}_R{round_number:02d}"

    def _get_dataset_hash(self) -> str:
        """Quick hash based on file count + total size in laps/ and race_results/ dirs."""
        total = 0
        count = 0
        for subdir in ["laps", "race_results", "weather"]:
            p = self.data_root / subdir
            if p.exists():
                for f in p.rglob("*.csv"):
                    total += f.stat().st_size
                    count += 1
        return hashlib.md5(f"{count}:{total}".encode()).hexdigest()[:12]

    def _ensure_dataset(self) -> pd.DataFrame:
        """Build or return cached dataset. Uses disk parquet cache for speed."""
        current_hash = self._get_dataset_hash()

        # 1. Try in-memory cache
        if self._dataset_cache is not None and self._dataset_hash == current_hash:
            return self._dataset_cache

        # 2. Try disk parquet cache
        cache_dir = self.data_root / "_dataset_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = cache_dir / f"walk_forward_dataset_{current_hash}.parquet"

        if parquet_path.exists():
            print(f"[WalkForward] Loading cached dataset from {parquet_path.name}...")
            t0 = time.time()
            dataset = pd.read_parquet(parquet_path)
            self._dataset_cache = dataset
            self._dataset_hash = current_hash
            print(f"[WalkForward] Loaded {len(dataset)} rows in {time.time() - t0:.1f}s")
            return dataset

        # 3. Build from CSVs (slow, but only once)
        print("[WalkForward] Building dataset from CSVs (first time, will be cached)...")
        t0 = time.time()

        from train_grid_position_model import build_driver_event_dataset
        dataset = build_driver_event_dataset(self.data_root)

        # Normalize types
        dataset["year"] = pd.to_numeric(dataset["year"], errors="coerce").astype("Int64")
        dataset["round"] = pd.to_numeric(dataset["round"], errors="coerce").astype("Int64")

        # Add race result targets
        dataset = self._add_race_targets(dataset)

        # Save to parquet for next time
        try:
            # Clean old caches
            for old in cache_dir.glob("walk_forward_dataset_*.parquet"):
                if old != parquet_path:
                    old.unlink(missing_ok=True)
            dataset.to_parquet(parquet_path, index=False)
            print(f"[WalkForward] Cached to {parquet_path.name}")
        except Exception as exc:
            print(f"[WalkForward] Warning: could not cache parquet: {exc}")

        self._dataset_cache = dataset
        self._dataset_hash = current_hash
        self._model_cache.clear()

        elapsed = time.time() - t0
        print(f"[WalkForward] Dataset built: {len(dataset)} rows in {elapsed:.1f}s")
        return dataset

    def _add_race_targets(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Add race result columns (position, points, DNF status) for training."""
        race_root = self.data_root / "race_results"
        if not race_root.exists():
            return dataset

        frames = []
        for f in sorted(race_root.rglob("*.csv")):
            try:
                df = pd.read_csv(f)
                if not df.empty:
                    frames.append(df)
            except Exception:
                continue

        if not frames:
            return dataset

        results = pd.concat(frames, ignore_index=True)
        results["year"] = pd.to_numeric(results["year"], errors="coerce").astype("Int64")
        results["round"] = pd.to_numeric(results["round"], errors="coerce").astype("Int64")

        # Use driver_code as the join key (matches the 3-letter code in the grid dataset)
        if "driver_code" in results.columns:
            # Drop the full name 'driver' column if it exists to avoid dups
            if "driver" in results.columns:
                results = results.drop(columns=["driver"])
            results = results.rename(columns={"driver_code": "driver"})

        # Build target columns
        results["position_num"] = pd.to_numeric(results.get("position", pd.NA), errors="coerce")
        results["points_num"] = pd.to_numeric(results.get("points", pd.NA), errors="coerce").fillna(0)

        classified = results.get("classified_position", results.get("position", pd.NA)).astype(str).str.strip().str.upper()
        results["dnf_target"] = (~classified.str.fullmatch(r"\d+")).astype(int)

        status_cat = results.get("status_category", pd.Series("", index=results.index)).astype(str).str.lower()
        results["accident_target"] = (status_cat == "accident").astype(int)
        results["mechanical_target"] = (status_cat == "mechanical failure").astype(int)

        results["winner_target"] = (results["position_num"] == 1).astype(int)
        results["podium_target"] = (results["position_num"] <= 3).astype(int)
        results["points_target"] = (results["points_num"] > 0).astype(int)

        # Merge - only add target columns that don't already exist
        target_cols = [
            "year", "round", "driver",
            "position_num", "points_num",
            "dnf_target", "accident_target", "mechanical_target",
            "winner_target", "podium_target", "points_target",
        ]
        target_cols = [c for c in target_cols if c in results.columns]
        merge_df = results[target_cols].drop_duplicates(subset=["year", "round", "driver"], keep="first")

        dataset = dataset.merge(merge_df, on=["year", "round", "driver"], how="left")
        return dataset

    def predict_with_retrain(self, season: int, round_number: int) -> dict[str, Any]:
        """Main entry point: retrain on data before (season, round), predict that race."""
        t0 = time.time()

        dataset = self._ensure_dataset()

        # Split: train = before target, test = target race
        train_mask = (
            (dataset["year"] < season)
            | ((dataset["year"] == season) & (dataset["round"] < round_number))
        )
        test_mask = (dataset["year"] == season) & (dataset["round"] == round_number)

        if test_mask.sum() == 0:
            raise ValueError(
                f"No feature data for season={season}, round={round_number}. "
                "Need to ingest practice/qualifying data first."
            )

        if train_mask.sum() < 20:
            raise ValueError(
                f"Insufficient training data: only {train_mask.sum()} rows before "
                f"season={season}, round={round_number}"
            )

        test_df = dataset.loc[test_mask].copy()
        event_name = str(test_df.iloc[0]["event_name"])
        if "team" not in test_df.columns:
            test_df["team"] = "UNKNOWN"
        test_df["team"] = test_df["team"].fillna("UNKNOWN")

        # Try cache
        cache_key = self._cache_key(season, round_number)
        cached = self._model_cache.get(cache_key)
        train_rows = int(train_mask.sum())

        if cached is not None:
            models = cached
        else:
            print(f"[WalkForward] Training models for {season} R{round_number:02d} ({train_rows} train rows)...")
            models = self._train_all_models(dataset, train_mask)
            self._model_cache[cache_key] = models

        # Predict
        predictions = self._run_inference(models, test_df, dataset, train_mask)

        elapsed = time.time() - t0

        # Get event date
        event_date = None
        try:
            import fastf1
            schedule = fastf1.get_event_schedule(season, include_testing=False)
            ev = schedule.loc[schedule["RoundNumber"] == round_number]
            if not ev.empty:
                d = pd.to_datetime(ev.iloc[0].get("EventDate"), errors="coerce")
                event_date = d.date().isoformat() if pd.notna(d) else None
        except Exception:
            pass

        has_weather = any(c.startswith("wx_") for c in test_df.columns if test_df[c].notna().any())
        has_sprint = "has_sprint" in test_df.columns and test_df["has_sprint"].max() > 0
        has_circuit = any(c.startswith("circuit_") for c in test_df.columns if test_df[c].notna().any())

        return {
            "season": int(season),
            "round": int(round_number),
            "event_name": event_name,
            "event_date": event_date,
            "feature_source": "walk_forward_retrain",
            "drivers": predictions,
            "retrain_info": {
                "training_rows": train_rows,
                "training_time_seconds": round(elapsed, 1),
                "train_cutoff": f"{season}-R{round_number - 1:02d}" if round_number > 1 else f"{season - 1}-last",
                "weather_features_used": has_weather,
                "sprint_features_used": has_sprint,
                "circuit_features_used": has_circuit,
            },
        }

    def _train_all_models(self, dataset: pd.DataFrame, train_mask: pd.Series) -> dict[str, Any]:
        """Train grid + classification models."""
        models: dict[str, Any] = {}

        exclude_targets = {
            "grid_position_target", "q_best_lap_seconds",
            "position_num", "points_num",
            "dnf_target", "accident_target", "mechanical_target",
            "winner_target", "podium_target", "points_target",
            "year", "round",
        }

        train_data = dataset.loc[train_mask]

        # 1. Grid position model
        if "grid_position_target" in train_data.columns and train_data["grid_position_target"].notna().sum() > 20:
            feature_cols, numeric, categorical = _get_feature_cols(train_data, exclude_targets)
            X = train_data[feature_cols]
            y = train_data["grid_position_target"].astype(float)
            model = _build_sklearn_pipeline(numeric, categorical, _regression_ensemble())
            model.fit(X, y)
            models["grid"] = {"model": model, "features": feature_cols}
            print("  [OK] Grid model trained")

        # 2. Points classifier
        if "points_target" in train_data.columns and train_data["points_target"].notna().sum() > 20:
            feature_cols, numeric, categorical = _get_feature_cols(train_data, exclude_targets)
            X = train_data[feature_cols]
            y = train_data["points_target"].astype(int)
            model = _build_sklearn_pipeline(numeric, categorical, _classification_ensemble())
            model.fit(X, y)
            models["points"] = {"model": model, "features": feature_cols}
            print("  [OK] Points model trained")

        # 3. DNF classifier
        if "dnf_target" in train_data.columns and train_data["dnf_target"].notna().sum() > 20:
            feature_cols, numeric, categorical = _get_feature_cols(train_data, exclude_targets)
            X = train_data[feature_cols]
            y = train_data["dnf_target"].astype(int)
            model = _build_sklearn_pipeline(numeric, categorical, _classification_ensemble())
            model.fit(X, y)
            models["dnf"] = {"model": model, "features": feature_cols}
            print("  [OK] DNF model trained")

        # 4. Outcome multiclass
        if all(c in train_data.columns for c in ["winner_target", "podium_target", "points_target", "dnf_target"]):
            dnf = train_data["dnf_target"].fillna(0).astype(int) == 1
            winner = train_data["winner_target"].fillna(0).astype(int) == 1
            podium = train_data["podium_target"].fillna(0).astype(int) == 1
            points = train_data["points_target"].fillna(0).astype(int) == 1
            outcome = np.where(dnf, "DNF", np.where(winner, "WIN", np.where(podium, "PODIUM", np.where(points, "POINTS", "NO_POINTS"))))

            feature_cols, numeric, categorical = _get_feature_cols(train_data, exclude_targets)
            X = train_data[feature_cols]
            model = _build_sklearn_pipeline(numeric, categorical, _classification_ensemble())
            model.fit(X, outcome)
            models["outcome"] = {"model": model, "features": feature_cols}
            print("  [OK] Outcome model trained")

        return models

    def _run_inference(self, models: dict[str, Any], test_df: pd.DataFrame, full_dataset: pd.DataFrame, train_mask: pd.Series) -> list[dict[str, Any]]:
        """Run prediction using retrained models."""
        n = len(test_df)

        # Grid model
        grid_info = models.get("grid")
        if grid_info:
            features = self._align_features(test_df, grid_info["features"])
            grid_scores = grid_info["model"].predict(features)
        else:
            grid_scores = np.arange(1, n + 1, dtype=float)

        # Points probability
        points_info = models.get("points")
        if points_info:
            features = self._align_features(test_df, points_info["features"])
            proba = points_info["model"].predict_proba(features)
            classes = list(points_info["model"].classes_)
            p_points = proba[:, classes.index(1)] if 1 in classes else np.full(n, 0.5)
        else:
            p_points = np.full(n, 0.5)

        # DNF probability
        dnf_info = models.get("dnf")
        if dnf_info:
            features = self._align_features(test_df, dnf_info["features"])
            proba = dnf_info["model"].predict_proba(features)
            classes = list(dnf_info["model"].classes_)
            p_dnf = proba[:, classes.index(1)] if 1 in classes else np.full(n, 0.1)
        else:
            p_dnf = np.full(n, 0.1)

        # Outcome probabilities
        outcome_info = models.get("outcome")
        p_win = np.zeros(n)
        p_podium = np.zeros(n)
        if outcome_info:
            features = self._align_features(test_df, outcome_info["features"])
            proba = outcome_info["model"].predict_proba(features)
            classes = [str(c).upper() for c in outcome_info["model"].classes_]

            if "WIN" in classes:
                p_win = proba[:, classes.index("WIN")]
            if "PODIUM" in classes:
                p_podium = p_win + proba[:, classes.index("PODIUM")]
            if "DNF" in classes:
                p_dnf_out = proba[:, classes.index("DNF")]
                p_dnf = np.clip(0.6 * p_dnf + 0.4 * p_dnf_out, 0, 1)

        p_podium = np.clip(p_podium, 0, 1)

        # Composite finish score
        norm = np.nanstd(grid_scores)
        norm = float(norm) if np.isfinite(norm) and norm > 0 else 1.0
        grid_signal = (grid_scores - np.nanmean(grid_scores)) / norm

        finish_score = (
            2.2 * p_win
            + 1.35 * p_podium
            + 0.9 * p_points
            - 1.15 * p_dnf
            + 0.15 * grid_signal
        )

        # Build output
        grid_pos = test_df.get("grid_position_target", test_df.get("official_grid_position", pd.NA))

        result = []
        for i, (idx, row) in enumerate(test_df.iterrows()):
            gp = grid_pos.iloc[i] if not isinstance(grid_pos, type(pd.NA)) else pd.NA
            result.append({
                "driver_code": str(row["driver"]),
                "driver_name": str(row["driver"]),
                "team": str(row.get("team", "UNKNOWN")),
                "grid_position": int(gp) if pd.notna(gp) else None,
                "p_win": round(float(p_win[i]), 4),
                "p_podium": round(float(p_podium[i]), 4),
                "p_points": round(float(p_points[i]), 4),
                "p_dnf": round(float(p_dnf[i]), 4),
                "_score": float(finish_score[i]),
            })

        result.sort(key=lambda x: (-x["_score"], x.get("grid_position") or 99))
        for rank, d in enumerate(result, 1):
            d["predicted_finish"] = rank
            del d["_score"]

        return result

    @staticmethod
    def _align_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        """Ensure df has exactly the expected feature columns."""
        out = df.copy()
        for col in feature_cols:
            if col not in out.columns:
                out[col] = pd.NA
        return out[feature_cols]
