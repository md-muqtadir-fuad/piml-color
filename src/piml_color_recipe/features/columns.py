from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd


def safe_numeric(series: pd.Series) -> pd.Series:
    """Convert a Series to numeric without raising on dirty industrial strings."""
    return pd.to_numeric(series, errors="coerce")


def sanitize_feature_name(name: object) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def nm_from_col(column: str) -> int:
    match = re.search(r"_(\d+)nm$", str(column))
    if not match:
        raise ValueError(f"No wavelength suffix found in column: {column}")
    return int(match.group(1))


def sorted_nm_columns(columns: Iterable[str], marker: str) -> list[str]:
    """Return wavelength columns sorted by nm.

    marker can be a full prefix (e.g. ``target_refl__reflectance``) or a suffix
    marker (e.g. ``reflectance``). This supports both old and src-first master
    column naming.
    """
    result: list[tuple[int, str]] = []
    for col in columns:
        c = str(col)
        if not c.endswith("nm"):
            continue
        if f"{marker}_" in c or c.startswith(f"{marker}_"):
            try:
                result.append((nm_from_col(c), c))
            except ValueError:
                pass
    return [c for _, c in sorted(result)]


def unique_existing(columns: Iterable[str], df: pd.DataFrame) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for col in columns:
        if col in df.columns and col not in seen:
            seen.add(col)
            out.append(col)
    return out
