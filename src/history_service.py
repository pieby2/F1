from __future__ import annotations

import ast
import hashlib
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from langchain_community.utilities import SQLDatabase
except Exception:  # pragma: no cover - optional dependency
    SQLDatabase = None


RESULT_TABLE = "results_history"
METADATA_TABLE = "history_metadata"
HISTORY_COLUMNS = [
    "year",
    "round",
    "event_name",
    "session_type",
    "driver_number",
    "driver_code",
    "driver",
    "team",
    "position",
    "classified_position",
    "grid_position",
    "status",
    "status_category",
    "points",
    "source",
]


class Formula1HistoryService:
    def __init__(self, data_root: Path | str = Path("data") / "fastf1_csv", cache_dir: Path | str | None = None) -> None:
        self.data_root = Path(data_root)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else self.data_root / "_api_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "history.sqlite"
        self._sql_database: SQLDatabase | None = None
        self._history_frame: pd.DataFrame | None = None
        self._available_seasons: list[int] = []
        self._latest_season: int | None = None
        self._ensure_database()

    def _database_uri(self) -> str:
        return f"sqlite:///{self.db_path.resolve().as_posix()}"

    def _source_signature(self) -> str:
        digest = hashlib.sha256()
        for folder in ("race_results", "sprint_results"):
            root = self.data_root / folder
            if not root.exists():
                continue
            for csv_file in sorted(root.rglob("*.csv")):
                stat = csv_file.stat()
                digest.update(str(csv_file.relative_to(self.data_root)).encode("utf-8"))
                digest.update(str(stat.st_size).encode("utf-8"))
                digest.update(str(int(stat.st_mtime)).encode("utf-8"))
        return digest.hexdigest()

    def _create_sql_database(self) -> SQLDatabase | None:
        if SQLDatabase is None:
            return None

        try:
            return SQLDatabase.from_uri(self._database_uri())
        except Exception:
            return None

    @staticmethod
    def _coerce_rows(result: Any) -> list[tuple[Any, ...]] | None:
        if result is None:
            return None

        if isinstance(result, str):
            text = result.strip()
            if not text or text.startswith("Error:"):
                return None
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                return None
            if isinstance(parsed, list):
                rows: list[tuple[Any, ...]] = []
                for item in parsed:
                    if isinstance(item, tuple):
                        rows.append(item)
                    elif isinstance(item, list):
                        rows.append(tuple(item))
                return rows
            return None

        if isinstance(result, list):
            rows: list[tuple[Any, ...]] = []
            for item in result:
                if isinstance(item, tuple):
                    rows.append(item)
                elif isinstance(item, list):
                    rows.append(tuple(item))
            return rows

        return None

    def _run_query(self, query: str) -> list[tuple[Any, ...]]:
        if self._sql_database is not None:
            rows = self._coerce_rows(self._sql_database.run_no_throw(query))
            if rows is not None:
                return rows

        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(query)
            return [tuple(row) for row in cursor.fetchall()]

    def _read_source_frame(self, root: Path, session_type: str) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        if not root.exists():
            return pd.DataFrame(columns=HISTORY_COLUMNS + ["source_file"])

        for csv_file in sorted(root.rglob("*.csv")):
            try:
                frame = pd.read_csv(csv_file)
            except Exception:
                continue

            if frame.empty:
                continue

            for column in HISTORY_COLUMNS:
                if column not in frame.columns:
                    frame[column] = pd.NA

            frame = frame[HISTORY_COLUMNS].copy()
            frame["session_type"] = session_type
            frame["source_file"] = str(csv_file.relative_to(self.data_root))
            frames.append(frame)

        if not frames:
            return pd.DataFrame(columns=HISTORY_COLUMNS + ["source_file"])

        combined = pd.concat(frames, ignore_index=True)
        for column in ["year", "round", "driver_number", "position", "classified_position", "grid_position", "points"]:
            combined[column] = pd.to_numeric(combined[column], errors="coerce")

        for column in ["event_name", "driver_code", "driver", "team", "status", "status_category", "source", "session_type", "source_file"]:
            combined[column] = combined[column].fillna("").astype(str)

        combined["year"] = pd.to_numeric(combined["year"], errors="coerce").astype("Int64")
        combined["round"] = pd.to_numeric(combined["round"], errors="coerce").astype("Int64")
        combined["driver_number"] = pd.to_numeric(combined["driver_number"], errors="coerce")
        combined["position"] = pd.to_numeric(combined["position"], errors="coerce")
        combined["classified_position"] = pd.to_numeric(combined["classified_position"], errors="coerce")
        combined["grid_position"] = pd.to_numeric(combined["grid_position"], errors="coerce")
        combined["points"] = pd.to_numeric(combined["points"], errors="coerce").fillna(0)

        sort_columns = ["year", "round", "session_type", "event_name", "driver_code"]
        combined = combined.sort_values(sort_columns).reset_index(drop=True)
        return combined

    def _ensure_database(self) -> None:
        signature = self._source_signature()
        if self.db_path.exists():
            try:
                with sqlite3.connect(self.db_path) as connection:
                    row = connection.execute(
                        f"SELECT value FROM {METADATA_TABLE} WHERE key = ?",
                        ("source_signature",),
                    ).fetchone()
                    latest = connection.execute(
                        f"SELECT value FROM {METADATA_TABLE} WHERE key = ?",
                        ("latest_season",),
                    ).fetchone()
                    if row and row[0] == signature:
                        self._latest_season = int(latest[0]) if latest and latest[0] else None
                        self._sql_database = self._create_sql_database()
                        return
            except Exception:
                pass

        race_frame = self._read_source_frame(self.data_root / "race_results", "race")
        sprint_frame = self._read_source_frame(self.data_root / "sprint_results", "sprint")
        combined = pd.concat([frame for frame in [race_frame, sprint_frame] if not frame.empty], ignore_index=True)
        if combined.empty:
            combined = pd.DataFrame(columns=HISTORY_COLUMNS + ["session_type", "source_file"])

        available_seasons = sorted({int(year) for year in combined["year"].dropna().tolist()}) if not combined.empty else []
        latest_season = available_seasons[-1] if available_seasons else None

        with sqlite3.connect(self.db_path) as connection:
            combined.to_sql(RESULT_TABLE, connection, if_exists="replace", index=False)
            connection.execute(
                f"CREATE TABLE IF NOT EXISTS {METADATA_TABLE} (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            metadata_rows = [
                ("source_signature", signature),
                ("latest_season", str(latest_season or "")),
                ("available_seasons", json.dumps(available_seasons)),
                ("row_count", str(len(combined))),
                ("updated_at", datetime.now(timezone.utc).isoformat()),
            ]
            connection.executemany(
                f"INSERT OR REPLACE INTO {METADATA_TABLE} (key, value) VALUES (?, ?)",
                metadata_rows,
            )
            connection.commit()

        self._sql_database = self._create_sql_database()
        self._history_frame = None
        self._available_seasons = available_seasons
        self._latest_season = latest_season

    def _load_history_frame(self) -> pd.DataFrame:
        if self._history_frame is not None:
            return self._history_frame

        query = (
            f"SELECT year, round, event_name, session_type, driver_number, driver_code, driver, team, "
            f"position, classified_position, grid_position, status, status_category, points, source "
            f"FROM {RESULT_TABLE}"
        )
        rows = self._run_query(query)
        if not rows:
            frame = pd.DataFrame(columns=HISTORY_COLUMNS)
        else:
            frame = pd.DataFrame(rows, columns=HISTORY_COLUMNS)

        for column in ["year", "round", "driver_number", "position", "classified_position", "grid_position", "points"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        for column in ["event_name", "driver_code", "driver", "team", "status", "status_category", "source", "session_type"]:
            frame[column] = frame[column].fillna("").astype(str)

        frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
        frame["round"] = pd.to_numeric(frame["round"], errors="coerce").astype("Int64")
        frame["points"] = pd.to_numeric(frame["points"], errors="coerce").fillna(0)
        frame["position"] = pd.to_numeric(frame["position"], errors="coerce")
        frame["classified_position"] = pd.to_numeric(frame["classified_position"], errors="coerce")
        frame["grid_position"] = pd.to_numeric(frame["grid_position"], errors="coerce")

        self._history_frame = frame.sort_values(["year", "round", "session_type", "event_name", "driver_code"]).reset_index(drop=True)

        if not self._available_seasons:
            self._available_seasons = sorted({int(year) for year in frame["year"].dropna().tolist()})
        if self._latest_season is None and self._available_seasons:
            self._latest_season = self._available_seasons[-1]

        return self._history_frame

    @staticmethod
    def _format_points(value: float) -> str:
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.1f}"

    def build_history_summary(self, season: int | None = None, limit: int = 20) -> dict[str, Any]:
        frame = self._load_history_frame()
        if frame.empty:
            raise RuntimeError("No race history data is available yet.")

        available_seasons = self._available_seasons or sorted({int(year) for year in frame["year"].dropna().tolist()})
        if not available_seasons:
            raise RuntimeError("No seasonal history could be derived from the cached database.")

        earliest_season = available_seasons[0]
        latest_season = self._latest_season or available_seasons[-1]
        requested_season = int(season) if season is not None else latest_season
        resolved_season = max(earliest_season, min(requested_season, latest_season))

        filtered = frame[frame["year"].astype("Int64") <= resolved_season].copy()
        race_rows = filtered[filtered["session_type"].str.lower() == "race"].copy()
        if race_rows.empty:
            race_rows = filtered.copy()

        race_rows["position_num"] = pd.to_numeric(race_rows["position"], errors="coerce")
        race_rows["points_num"] = pd.to_numeric(race_rows["points"], errors="coerce").fillna(0)
        race_rows["finished"] = race_rows["status_category"].str.lower().eq("finished")

        latest_team_rows = race_rows.sort_values(["year", "round", "event_name", "driver_code"]).drop_duplicates(
            subset=["driver_code"], keep="last"
        )
        latest_team_by_driver = {
            str(row.driver_code): str(row.team)
            for row in latest_team_rows.itertuples(index=False)
            if str(row.driver_code).strip()
        }

        latest_name_rows = race_rows.sort_values(["year", "round", "event_name", "driver_code"]).drop_duplicates(
            subset=["driver_code"], keep="last"
        )
        latest_name_by_driver = {
            str(row.driver_code): str(row.driver)
            for row in latest_name_rows.itertuples(index=False)
            if str(row.driver_code).strip()
        }

        latest_season_rows = race_rows[race_rows["year"] == resolved_season]
        if latest_season_rows.empty:
            latest_season_rows = race_rows

        teammate_wins: defaultdict[str, int] = defaultdict(int)
        teammate_losses: defaultdict[str, int] = defaultdict(int)
        for (_, _, _, team), group in race_rows.groupby(["year", "round", "event_name", "team"], dropna=False):
            drivers = group.dropna(subset=["driver_code"]).copy()
            drivers["position_num"] = pd.to_numeric(drivers["position"], errors="coerce")
            if len(drivers) < 2:
                continue

            for left, right in combinations(drivers.itertuples(index=False), 2):
                left_pos = getattr(left, "position_num", pd.NA)
                right_pos = getattr(right, "position_num", pd.NA)
                if pd.isna(left_pos) or pd.isna(right_pos) or float(left_pos) == float(right_pos):
                    continue
                if float(left_pos) < float(right_pos):
                    teammate_wins[str(left.driver_code)] += 1
                    teammate_losses[str(right.driver_code)] += 1
                else:
                    teammate_wins[str(right.driver_code)] += 1
                    teammate_losses[str(left.driver_code)] += 1

        driver_rows: list[dict[str, Any]] = []
        for driver_code, group in race_rows.groupby("driver_code", dropna=False):
            code = str(driver_code).strip()
            if not code:
                continue

            ordered = group.sort_values(["year", "round", "event_name"])
            latest = ordered.iloc[-1]
            races = int(len(ordered))
            wins = int((ordered["position_num"] == 1).sum())
            podiums = int((ordered["position_num"] <= 3).sum())
            points = float(ordered["points_num"].sum())
            dnfs = int((~ordered["finished"]).sum())
            finish_values = ordered["position_num"].dropna()
            avg_finish = float(finish_values.mean()) if not finish_values.empty else None
            best_finish = int(finish_values.min()) if not finish_values.empty else None
            losses = max(races - wins, 0)
            teammate_win_count = teammate_wins.get(code, 0)
            teammate_loss_count = teammate_losses.get(code, 0)
            team = latest_team_by_driver.get(code, str(latest.get("team", "UNKNOWN"))) or "UNKNOWN"
            driver_name = latest_name_by_driver.get(code, str(latest.get("driver", code))) or code

            driver_rows.append(
                {
                    "driver_code": code,
                    "driver_name": driver_name,
                    "team": team,
                    "races": races,
                    "wins": wins,
                    "losses": losses,
                    "podiums": podiums,
                    "points": round(points, 1),
                    "dnfs": dnfs,
                    "avg_finish": round(avg_finish, 2) if avg_finish is not None else None,
                    "best_finish": best_finish,
                    "teammate_wins": teammate_win_count,
                    "teammate_losses": teammate_loss_count,
                    "summary": (
                        f"{driver_name} has {wins} wins, {podiums} podiums and {self._format_points(points)} points "
                        f"for {team}. Team-mate record: {teammate_win_count}-{teammate_loss_count}."
                    ),
                }
            )

        team_rows: list[dict[str, Any]] = []
        team_roster = latest_season_rows.dropna(subset=["team", "driver_code"]).copy()
        team_roster["driver_name"] = team_roster["driver_code"].map(latest_name_by_driver).fillna(team_roster["driver"])
        roster_by_team = {
            str(team): sorted({str(name) for name in group["driver_name"].tolist() if str(name).strip()})
            for team, group in team_roster.groupby("team", dropna=False)
        }

        for team_name, group in race_rows.groupby("team", dropna=False):
            team = str(team_name).strip() or "UNKNOWN"
            ordered = group.sort_values(["year", "round", "event_name", "driver_code"])
            races = int(len(ordered))
            wins = int((ordered["position_num"] == 1).sum())
            podiums = int((ordered["position_num"] <= 3).sum())
            points = float(ordered["points_num"].sum())
            dnfs = int((~ordered["finished"]).sum())
            finish_values = ordered["position_num"].dropna()
            avg_finish = float(finish_values.mean()) if not finish_values.empty else None
            best_finish = int(finish_values.min()) if not finish_values.empty else None
            losses = max(races - wins, 0)

            team_rows.append(
                {
                    "team": team,
                    "races": races,
                    "wins": wins,
                    "losses": losses,
                    "podiums": podiums,
                    "points": round(points, 1),
                    "dnfs": dnfs,
                    "avg_finish": round(avg_finish, 2) if avg_finish is not None else None,
                    "best_finish": best_finish,
                    "drivers": roster_by_team.get(team, []),
                    "summary": (
                        f"{team} has {wins} wins, {podiums} podiums and {self._format_points(points)} points "
                        f"across the available race history."
                    ),
                }
            )

        driver_rows.sort(key=lambda item: (-item["wins"], -item["podiums"], -item["points"], item["driver_code"]))
        team_rows.sort(key=lambda item: (-item["wins"], -item["podiums"], -item["points"], item["team"]))

        if limit > 0:
            driver_rows = driver_rows[:limit]
            team_rows = team_rows[:limit]

        top_driver = driver_rows[0] if driver_rows else None
        top_team = team_rows[0] if team_rows else None
        history_note = (
            f"Showing race history through {resolved_season}."
            if resolved_season == latest_season
            else f"Requested season {requested_season}, but the local database currently stops at {latest_season}; showing history through {resolved_season}."
        )

        overall_summary_parts = [history_note]
        highlights = [history_note]
        if top_driver is not None:
            overall_summary_parts.append(
                f"{top_driver['driver_name']} leads the driver table with {top_driver['wins']} wins and {top_driver['podiums']} podiums."
            )
            highlights.append(top_driver["summary"])
        if top_team is not None:
            overall_summary_parts.append(
                f"{top_team['team']} leads the constructors table with {top_team['wins']} wins and {top_team['podiums']} podiums."
            )
            highlights.append(top_team["summary"])

        return {
            "requested_season": requested_season,
            "resolved_season": resolved_season,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "availability_note": history_note,
            "available_seasons": available_seasons,
            "overall_summary": " ".join(overall_summary_parts),
            "highlights": highlights,
            "drivers": driver_rows,
            "teams": team_rows,
        }