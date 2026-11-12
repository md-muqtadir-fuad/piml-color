from __future__ import annotations

import re
from typing import Any

import pandas as pd

PANTONE_RE = re.compile(r"\bPANTONE\b\s+\d{2}-\d{4}\s+TCX\b", flags=re.IGNORECASE)
SAMPLE_RE = re.compile(r"^\s*[0-9]{2}\s*[kK]\s*[0-9]{3,6}\s*[A-Za-z]\s*$")
RECIPE_RE = re.compile(r"^\s*[0-9]{2}\s*[kK]\s*[0-9]{3,6}\s*$")


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = text.replace("\ufeff", "")
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_id(value: Any) -> str:
    return clean_text(value).lower().replace(" ", "")


def normalize_code(value: Any) -> str:
    return normalize_id(value)


def normalize_target_id(value: Any) -> str:
    text = clean_text(value).upper()
    text = text.replace("–", "-").replace("—", "-").replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.replace("PANTONEPANTONE", "PANTONE")
    return text


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [clean_text(c).strip().replace(" ", "_") for c in out.columns]
    return out


def looks_like_pantone(value: Any) -> bool:
    return bool(PANTONE_RE.search(clean_text(value)))


def looks_like_sample(value: Any) -> bool:
    return bool(SAMPLE_RE.match(clean_text(value)))


def looks_like_recipe(value: Any) -> bool:
    return bool(RECIPE_RE.match(clean_text(value)))


def to_numeric_value(value: Any) -> float:
    if pd.isna(value):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = clean_text(value)
    if text == "":
        return float("nan")
    text = text.replace("%", "").replace(",", ".")
    text = re.sub(r"[^0-9eE+\-.]", "", text)
    if text in {"", ".", "-", "+"}:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def to_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
        .replace({"": None, "nan": None, "None": None}),
        errors="coerce",
    )
