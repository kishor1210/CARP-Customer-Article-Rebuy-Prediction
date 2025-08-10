from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from .config import Columns


FEATURE_COLUMNS_DEFAULT = [
    "order_index",
    "avg_days_between_orders",
    "avg_basket_size",
    "unique_articles_so_far",
    "preferred_dow",
    "times_bought_so_far",
    "avg_gap_days",
    "days_since_last_purchase",
    "fraction_orders_with_article",
    "article_popularity",
    "article_median_repurchase_days",
    "article_peak_month",
]


def prepare_train_valid(dataset: pd.DataFrame, train_end_date: str, feature_cols: List[str] | None = None) -> Tuple[xgb.DMatrix, xgb.DMatrix, List[str]]:
    feature_cols = feature_cols or FEATURE_COLUMNS_DEFAULT
    df = dataset.copy()
    df = df.dropna(subset=[Columns.order_date])
    df[Columns.order_date] = pd.to_datetime(df[Columns.order_date])
    mask_train = df[Columns.order_date] <= pd.to_datetime(train_end_date)
    train_df = df[mask_train]
    valid_df = df[~mask_train]

    X_train = train_df[feature_cols]
    y_train = train_df["label"].astype(int)
    X_valid = valid_df[feature_cols]
    y_valid = valid_df["label"].astype(int)

    return xgb.DMatrix(X_train, label=y_train), xgb.DMatrix(X_valid, label=y_valid), feature_cols


def train_xgb(train_dmat: xgb.DMatrix, valid_dmat: xgb.DMatrix) -> xgb.Booster:
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "logloss"],
        "tree_method": "hist",
        "max_depth": 7,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1.0,
        "lambda": 1.0,
    }
    evals = [(train_dmat, "train"), (valid_dmat, "valid")]
    bst = xgb.train(params, train_dmat, num_boost_round=400, evals=evals, early_stopping_rounds=30, verbose_eval=False)
    return bst


def evaluate_auc(bst: xgb.Booster, valid_dmat: xgb.DMatrix) -> float:
    y_true = valid_dmat.get_label()
    y_pred = bst.predict(valid_dmat)
    return float(roc_auc_score(y_true, y_pred))


def predict_proba(bst: xgb.Booster, feature_frame: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    dmat = xgb.DMatrix(feature_frame[feature_cols])
    probs = bst.predict(dmat)
    out = feature_frame[[Columns.customer_id, Columns.article_id, Columns.order_date]].copy()
    out["prob_rebuy"] = probs
    return out


