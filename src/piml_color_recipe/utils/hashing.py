from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def make_file_manifest(paths: Iterable[str | Path], root: str | Path | None = None) -> pd.DataFrame:
    root_path = Path(root).resolve() if root is not None else None
    rows: list[dict[str, object]] = []
    for p in paths:
        path = Path(p).resolve()
        exists = path.exists()
        rel = str(path.relative_to(root_path)) if root_path and exists and path.is_relative_to(root_path) else str(path)
        rows.append(
            {
                "path": rel,
                "exists": bool(exists),
                "size_bytes": int(path.stat().st_size) if exists else None,
                "sha256": file_sha256(path) if exists and path.is_file() else None,
            }
        )
    return pd.DataFrame(rows)
