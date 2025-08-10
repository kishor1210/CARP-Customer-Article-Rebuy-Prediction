from __future__ import annotations

import pandas as pd

from .config import Columns
from .utils import ensure_datetime


def read_transactions_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {Columns.customer_id, Columns.article_id, Columns.order_date, Columns.quantity}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df[Columns.order_date] = ensure_datetime(df[Columns.order_date])
    # enforce dtypes
    df[Columns.customer_id] = df[Columns.customer_id].astype(str)
    df[Columns.article_id] = df[Columns.article_id].astype(str)
    df[Columns.quantity] = pd.to_numeric(df[Columns.quantity], errors="coerce").fillna(0).astype(int)
    return df


