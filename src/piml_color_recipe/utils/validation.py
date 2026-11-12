from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Any

import pandas as pd


@dataclass
class SelfCheckResult:
    stage: str
    checks: dict[str, int | float | str | bool | list[Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def add_count(self, name: str, value: int | float | bool | str | list[Any]) -> None:
        self.checks[name] = value

    def warn_if(self, condition: bool, message: str) -> None:
        if condition:
            self.warnings.append(message)

    def fail_if(self, condition: bool, message: str) -> None:
        if condition:
            self.failures.append(message)

    @property
    def ok(self) -> bool:
        return len(self.failures) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "ok": self.ok,
            "checks": self.checks,
            "warnings": self.warnings,
            "failures": self.failures,
        }


def require_columns(df: pd.DataFrame, columns: Iterable[str], table_name: str = "DataFrame") -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{table_name} missing required columns: {missing}")


def assert_no_failures(result: SelfCheckResult) -> None:
    if result.failures:
        raise AssertionError(f"{result.stage} self-check failed: {result.failures}")


def duplicate_count(df: pd.DataFrame, subset: list[str]) -> int:
    if df.empty:
        return 0
    return int(df.duplicated(subset=subset, keep=False).sum())


def null_count(df: pd.DataFrame, column: str) -> int:
    return int(df[column].isna().sum()) if column in df.columns else -1
