from __future__ import annotations

import json
from typing import Iterable, List

import numpy as np
import pandas as pd


def ensure_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def days_between(d1: pd.Series, d2: pd.Series) -> pd.Series:
    return (d2 - d1).dt.days.astype("float32")


def safe_divide(numerator: pd.Series | float, denominator: pd.Series | float) -> pd.Series:
    return numerator / denominator.replace({0: np.nan})


def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def top_n_by_group(df: pd.DataFrame, group_keys: List[str], score_col: str, n: int) -> pd.DataFrame:
    ranked = df.sort_values(group_keys + [score_col], ascending=[True] * len(group_keys) + [False])
    return ranked.groupby(group_keys, as_index=False).head(n)


