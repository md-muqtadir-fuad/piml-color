from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved project directories used by all scripts."""

    root: Path
    raw: Path
    interim: Path
    processed: Path
    splits: Path
    manifests: Path
    outputs: Path
    logs: Path
    text_logs: Path
    csv_logs: Path
    json_run_summaries: Path
    models: Path
    reports: Path

    def ensure_base_dirs(self) -> None:
        for p in [
            self.raw,
            self.interim,
            self.processed,
            self.splits,
            self.manifests,
            self.outputs,
            self.logs,
            self.text_logs,
            self.csv_logs,
            self.json_run_summaries,
            self.models,
            self.reports,
        ]:
            p.mkdir(parents=True, exist_ok=True)

    def output_stage_dir(self, stage: str) -> Path:
        p = self.outputs / stage
        p.mkdir(parents=True, exist_ok=True)
        (p / "tables").mkdir(parents=True, exist_ok=True)
        (p / "figures").mkdir(parents=True, exist_ok=True)
        return p


def find_project_root(start: str | Path | None = None) -> Path:
    """Find project root by walking upward from start/current file location."""
    if start is None:
        start_path = Path.cwd().resolve()
    else:
        start_path = Path(start).resolve()
        if start_path.is_file():
            start_path = start_path.parent

    candidates = [start_path] + list(start_path.parents)
    for candidate in candidates:
        if (candidate / "pyproject.toml").exists() and (candidate / "data").exists():
            return candidate
        if (candidate / "data" / "raw").exists() and (candidate / "scripts").exists():
            return candidate
    return start_path


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists() or yaml is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    return data


def _get(mapping: Mapping[str, Any], dotted: str, default: str) -> str:
    cur: Any = mapping
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return str(cur)


def load_paths(project_root: str | Path | None = None) -> ProjectPaths:
    root = find_project_root(project_root)
    cfg = _read_yaml(root / "configs" / "paths.yaml")

    def rel(dotted: str, default: str) -> Path:
        value = _get(cfg, dotted, default)
        path = Path(value)
        return path if path.is_absolute() else root / path

    paths = ProjectPaths(
        root=root,
        raw=rel("data.raw", "data/raw"),
        interim=rel("data.interim", "data/interim"),
        processed=rel("data.processed", "data/processed"),
        splits=rel("data.splits", "data/splits"),
        manifests=rel("data.manifests", "data/manifests"),
        outputs=rel("outputs.root", "outputs"),
        logs=rel("outputs.logs", "outputs/logs"),
        text_logs=rel("outputs.text_logs", "outputs/logs/text_logs"),
        csv_logs=rel("outputs.csv_logs", "outputs/logs/csv_logs"),
        json_run_summaries=rel("outputs.json_run_summaries", "outputs/logs/json_run_summaries"),
        models=rel("models", "models"),
        reports=rel("reports", "reports"),
    )
    paths.ensure_base_dirs()
    return paths
