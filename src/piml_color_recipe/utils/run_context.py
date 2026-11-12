from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .io import append_csv_row, write_csv, write_json
from .logging_utils import setup_logger
from .paths import ProjectPaths, load_paths
from .validation import SelfCheckResult


@dataclass
class PipelineRunContext:
    """Shared run context for final reproducible scripts.

    It centralizes:
    - resolved paths
    - text logging
    - CSV run/error/warning/self-check logs
    - stage table/figure directories
    - run-summary JSON output
    """

    script_name: str
    stage: str
    project_root: str | Path | None = None
    script_version: str = "unversioned"
    paths: ProjectPaths = field(init=False)
    run_id: str = field(init=False)
    started_at: str = field(init=False)
    log_path: Path = field(init=False)
    logger: logging.Logger = field(init=False)
    stage_dir: Path = field(init=False)
    tables_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)
    _t0: float = field(default_factory=time.time, init=False)

    def __post_init__(self) -> None:
        self.paths = load_paths(self.project_root)
        self.started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.stage_dir = self.paths.output_stage_dir(self.stage)
        self.tables_dir = self.stage_dir / "tables"
        self.figures_dir = self.stage_dir / "figures"
        self.log_path = self.paths.text_logs / f"{self.script_name}_{self.run_id}.log"
        self.logger = setup_logger(self.script_name, self.log_path)
        self.logger.info("Started %s", self.script_name)
        self.logger.info("SCRIPT_VERSION: %s", self.script_version)
        self.logger.info("Project root: %s", self.paths.root)
        self.logger.info("Log path: %s", self.log_path)
        self._append_script_row(status="STARTED")

    def table_path(self, name: str) -> Path:
        return self.tables_dir / name

    def figure_path(self, name: str) -> Path:
        return self.figures_dir / name

    def write_df(self, df: pd.DataFrame, path: str | Path) -> Path:
        path = write_csv(df, path)
        self.logger.info("Wrote %s rows x %s cols -> %s", len(df), len(df.columns), path)
        return path

    def write_json(self, obj: Any, path: str | Path) -> Path:
        path = write_json(obj, path)
        self.logger.info("Wrote JSON -> %s", path)
        return path

    def log_warning(self, message: str, **extra: Any) -> None:
        self.logger.warning(message)
        row = self._base_row(status="WARNING") | {"message": message} | extra
        append_csv_row(row, self.paths.csv_logs / "warning_log.csv")

    def log_error(self, message: str, **extra: Any) -> None:
        self.logger.error(message)
        row = self._base_row(status="ERROR") | {"message": message} | extra
        append_csv_row(row, self.paths.csv_logs / "error_log.csv")

    def log_self_check(self, result: SelfCheckResult) -> None:
        row = self._base_row(status="OK" if result.ok else "FAILED") | result.to_dict()
        append_csv_row(row, self.paths.csv_logs / "self_check_summary.csv")
        self.logger.info("Self-check result: %s", result.to_dict())

    def finish(self, status: str = "OK", summary: dict[str, Any] | None = None) -> Path:
        elapsed = round(time.time() - self._t0, 3)
        payload = {
            "script_name": self.script_name,
            "script_version": self.script_version,
            "stage": self.stage,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "elapsed_seconds": elapsed,
            "status": status,
            "log_path": str(self.log_path),
            "summary": summary or {},
        }
        summary_path = self.table_path(f"{self.stage}_run_summary.json")
        self.write_json(payload, summary_path)
        self.write_json(payload, self.paths.json_run_summaries / f"{self.script_name}_{self.run_id}.json")
        self._append_script_row(status=status, elapsed_seconds=elapsed)
        self.logger.info("Finished %s with status=%s in %.3fs", self.script_name, status, elapsed)
        return summary_path

    def _base_row(self, status: str) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "script_name": self.script_name,
            "script_version": self.script_version,
            "stage": self.stage,
            "status": status,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _append_script_row(self, status: str, elapsed_seconds: float | None = None) -> None:
        row = self._base_row(status=status)
        row["elapsed_seconds"] = elapsed_seconds
        row["log_path"] = str(self.log_path)
        append_csv_row(row, self.paths.csv_logs / "script_run_summary.csv")
