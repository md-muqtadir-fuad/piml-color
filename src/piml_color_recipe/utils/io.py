from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

READ_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin1")


def read_csv_flexible(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read CSV with robust encoding fallback and consistent low_memory=False."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    last_error: Exception | None = None
    for enc in READ_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
        except UnicodeDecodeError as e:
            last_error = e
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Could not decode {path}: {last_error}")


def write_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding="utf-8-sig")
    return path


def write_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, default=_json_default)
    return path


def read_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def append_csv_row(row: dict[str, Any], path: str | Path, columns: Iterable[str] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if columns is not None:
        df = df.reindex(columns=list(columns))
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False, encoding="utf-8-sig")
    return path
