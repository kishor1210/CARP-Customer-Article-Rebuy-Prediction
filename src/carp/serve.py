from __future__ import annotations

from typing import List

import pandas as pd
import xgboost as xgb

from .config import Columns
from .feature_engineering import assemble_features
from .model import FEATURE_COLUMNS_DEFAULT, predict_proba
from .utils import read_json, top_n_by_group


def generate_recommendations(transactions: pd.DataFrame, model_path: str, feature_cols_path: str, top_n: int = 5) -> pd.DataFrame:
    features = assemble_features(transactions)
    feature_cols: List[str] = read_json(feature_cols_path)
    bst = xgb.Booster()
    bst.load_model(model_path)
    scored = predict_proba(bst, features, feature_cols)

    # pick last order per customer as reference time and select top N articles
    last_orders = (
        transactions[[Columns.customer_id, Columns.order_date]].drop_duplicates().sort_values([Columns.customer_id, Columns.order_date])
        .groupby(Columns.customer_id).tail(1)
    )
    latest_scores = scored.merge(last_orders, on=[Columns.customer_id, Columns.order_date], how="inner")
    topn = top_n_by_group(latest_scores, [Columns.customer_id], "prob_rebuy", top_n)
    return topn


