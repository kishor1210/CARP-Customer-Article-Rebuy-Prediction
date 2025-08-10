from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Columns
from .utils import ensure_datetime, days_between, safe_divide


def _asof_merge_by(left: pd.DataFrame, right: pd.DataFrame, by: str, on: str,
                   direction: str = "backward") -> pd.DataFrame:
    """Helper to perform merge_asof with required sorting and clean types (single by)."""
    left = left.copy()
    right = right.copy()
    left[on] = ensure_datetime(left[on])
    right[on] = ensure_datetime(right[on])
    # Drop NaT keys to avoid pandas merge_asof errors
    left = left.dropna(subset=[on])
    right = right.dropna(subset=[on])
    # Sort by key primarily, then group columns using stable mergesort (ensures global monotonic 'on')
    sort_cols = [on, by]
    left = left.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    right = right.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    try:
        return pd.merge_asof(
            left,
            right,
            by=by,
            on=on,
            direction=direction,
            allow_exact_matches=True,
        )
    except ValueError as err:
        # Attach debug info
        msg = f"merge_asof failed (by={by}, on={on}). Left head: {left[sort_cols].head().to_dict(orient='list')} Right head: {right[sort_cols].head().to_dict(orient='list')}\nOriginal error: {err}"
        raise ValueError(msg)


def _asof_merge_by_multi(left: pd.DataFrame, right: pd.DataFrame, by_cols: list[str], on: str,
                         direction: str = "backward") -> pd.DataFrame:
    """merge_asof for multiple `by` keys with robust sorting and NaT handling."""
    left = left.copy()
    right = right.copy()
    left[on] = ensure_datetime(left[on])
    right[on] = ensure_datetime(right[on])
    left = left.dropna(subset=[on])
    right = right.dropna(subset=[on])
    # Sort by key primarily, then group columns to satisfy merge_asof global monotonicity
    sort_cols = [on] + by_cols
    left = left.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    right = right.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    try:
        return pd.merge_asof(
            left,
            right,
            by=by_cols,
            on=on,
            direction=direction,
            allow_exact_matches=True,
        )
    except ValueError as err:
        msg = f"merge_asof failed (by={by_cols}, on={on}). Left head: {left[sort_cols].head().to_dict(orient='list')} Right head: {right[sort_cols].head().to_dict(orient='list')}\nOriginal error: {err}"
        raise ValueError(msg)


def _asof_merge_by_multi_grouped(left: pd.DataFrame, right: pd.DataFrame, by_cols: list[str], on: str,
                                 direction: str = "backward") -> pd.DataFrame:
    """Fallback merge_asof executed per group to avoid global sort constraints.

    This is slower but robust for moderate datasets and debugging.
    """
    left = left.copy()
    right = right.copy()
    left[on] = ensure_datetime(left[on])
    right[on] = ensure_datetime(right[on])

    # Prepare result collector
    merged_parts: list[pd.DataFrame] = []

    # Iterate over groups present in left
    for key_vals, left_g in left.groupby(by_cols, sort=False):
        # Normalize key tuple
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        mask = pd.Series(True, index=right.index)
        for col, val in zip(by_cols, key_vals):
            mask &= right[col] == val
        right_g = right.loc[mask]
        if right_g.empty:
            merged_parts.append(left_g)
            continue
        # Sort by 'on' and drop by_cols from right to avoid duplicate columns
        left_g = left_g.sort_values(on)
        right_g = right_g.sort_values(on).drop(columns=[c for c in by_cols if c in right_g.columns])
        merged_g = pd.merge_asof(left_g, right_g, on=on, direction=direction, allow_exact_matches=True)
        merged_parts.append(merged_g)

    # Concatenate and restore original order by on + by for determinism
    out = pd.concat(merged_parts, axis=0, ignore_index=True)
    out = out.sort_values([on] + by_cols).reset_index(drop=True)
    return out

def build_order_level_aggregates(transactions: pd.DataFrame) -> pd.DataFrame:
    # One row per (customer, order_date) with basket size
    order_baskets = (
        transactions
        .groupby([Columns.customer_id, Columns.order_date], as_index=False)
        .agg(basket_size=(Columns.quantity, "sum"), num_lines=(Columns.article_id, "nunique"))
    )
    order_baskets = order_baskets.sort_values([Columns.customer_id, Columns.order_date])
    return order_baskets


def compute_customer_features(order_baskets: pd.DataFrame) -> pd.DataFrame:
    df = order_baskets.copy()
    # prior orders so far
    df["order_index"] = df.groupby(Columns.customer_id).cumcount() + 1

    # days since previous order and average
    df["prev_order_date"] = df.groupby(Columns.customer_id)[Columns.order_date].shift(1)
    df["days_since_prev_order"] = days_between(df["prev_order_date"], df[Columns.order_date])
    df["avg_days_between_orders"] = (
        df.groupby(Columns.customer_id)["days_since_prev_order"].expanding().mean().reset_index(level=0, drop=True)
    )

    # rolling averages (up to previous order) for basket size
    df["cum_orders"] = df.groupby(Columns.customer_id)["order_index"].transform("max")
    df["avg_basket_size"] = (
        df.groupby(Columns.customer_id)["basket_size"].expanding().mean().reset_index(level=0, drop=True)
    )

    # unique articles so far
    # build from transactions
    # join back later at order_date
    return df


def compute_unique_articles_so_far(transactions: pd.DataFrame) -> pd.DataFrame:
    # Count unique articles a customer has bought up to (and including) an order_date.
    # Compute first time each (customer, article) was purchased, then cumulative count of first occurrences by date.
    firsts = (
        transactions[[Columns.customer_id, Columns.article_id, Columns.order_date]]
        .sort_values([Columns.customer_id, Columns.article_id, Columns.order_date])
        .groupby([Columns.customer_id, Columns.article_id], as_index=False)
        .first()
        .rename(columns={Columns.order_date: "first_article_date"})
    )
    daily_new = (
        firsts.groupby([Columns.customer_id, "first_article_date"], as_index=False)
        .size()
        .rename(columns={"size": "new_unique_count", "first_article_date": Columns.order_date})
    )
    daily_new = daily_new.sort_values([Columns.customer_id, Columns.order_date])
    daily_new["unique_articles_so_far"] = (
        daily_new.groupby(Columns.customer_id)["new_unique_count"].cumsum()
    )
    per_day = daily_new[[Columns.customer_id, Columns.order_date, "unique_articles_so_far"]]
    return per_day


def compute_customer_dow_preference(order_baskets: pd.DataFrame) -> pd.DataFrame:
    df = order_baskets.copy()
    df = df.sort_values([Columns.customer_id, Columns.order_date])
    df["dow"] = df[Columns.order_date].dt.dayofweek
    # one-hot encode dow 0..6
    for d in range(7):
        df[f"dow_{d}"] = (df["dow"] == d).astype(int)
    # cumulative counts per customer
    cum_cols = [f"dow_{d}" for d in range(7)]
    df[cum_cols] = df.groupby(Columns.customer_id)[cum_cols].cumsum()
    # preferred dow is argmax across cumulative counts at this date
    df["preferred_dow"] = df[cum_cols].idxmax(axis=1).str.replace("dow_", "", regex=False).astype(int)
    out = df[[Columns.customer_id, Columns.order_date, "preferred_dow"]]
    return out


def compute_article_features(transactions: pd.DataFrame) -> pd.DataFrame:
    df = transactions.copy()
    df = df.sort_values([Columns.article_id, Columns.order_date])

    # popularity up to date: cumulative unique customers based on first time each customer bought the article
    firsts = (
        df[[Columns.article_id, Columns.customer_id, Columns.order_date]]
        .sort_values([Columns.article_id, Columns.customer_id, Columns.order_date])
        .groupby([Columns.article_id, Columns.customer_id], as_index=False)
        .first()
        .rename(columns={Columns.order_date: "first_buy_date"})
    )
    daily_new = (
        firsts.groupby([Columns.article_id, "first_buy_date"], as_index=False)
        .size()
        .rename(columns={"size": "new_unique_customers", "first_buy_date": Columns.order_date})
    )
    daily_new = daily_new.sort_values([Columns.article_id, Columns.order_date])
    daily_new["article_popularity"] = (
        daily_new.groupby(Columns.article_id)["new_unique_customers"].cumsum()
    )
    popularity_curve = daily_new[[Columns.article_id, Columns.order_date, "article_popularity"]]

    # rough repurchase cycle: median days between purchases per article
    art_order_days = (
        df.groupby([Columns.article_id, Columns.customer_id])[Columns.order_date]
        .apply(lambda s: s.sort_values()).reset_index(name=Columns.order_date)
    )
    art_order_days["prev_date"] = art_order_days.groupby([Columns.article_id, Columns.customer_id])[Columns.order_date].shift(1)
    gaps = art_order_days.dropna(subset=["prev_date"]).copy()
    gaps["gap_days"] = days_between(gaps["prev_date"], gaps[Columns.order_date])
    repurchase_cycle = (
        gaps.groupby(Columns.article_id)["gap_days"].median().rename("article_median_repurchase_days").reset_index()
    )

    # seasonal peak: most frequent month
    df["month"] = df[Columns.order_date].dt.month
    peak_month = (
        df.groupby(Columns.article_id)["month"].agg(lambda s: s.value_counts().idxmax()).rename("article_peak_month").reset_index()
    )

    # consolidate: popularity is time-varying, others are static per article
    art_static = repurchase_cycle.merge(peak_month, on=Columns.article_id, how="outer")
    art_features = popularity_curve.merge(art_static, on=Columns.article_id, how="left")
    return art_features


def compute_interaction_features(transactions: pd.DataFrame) -> pd.DataFrame:
    tx = transactions.copy()
    tx = tx.sort_values([Columns.customer_id, Columns.article_id, Columns.order_date])

    # times bought so far, last purchase date, average gap
    tx["purchase_index"] = tx.groupby([Columns.customer_id, Columns.article_id]).cumcount() + 1
    tx["prev_date"] = tx.groupby([Columns.customer_id, Columns.article_id])[Columns.order_date].shift(1)
    tx["gap_days"] = days_between(tx["prev_date"], tx[Columns.order_date])

    avg_gap = (
        tx.groupby([Columns.customer_id, Columns.article_id])["gap_days"].expanding().mean().reset_index(level=[0,1], drop=True)
    )
    tx["avg_gap_days"] = avg_gap

    # fraction of orders containing this article (up to current date)
    orders_per_cust = (
        tx[[Columns.customer_id, Columns.order_date]].drop_duplicates()
        .sort_values([Columns.customer_id, Columns.order_date])
    )
    orders_per_cust["order_number"] = orders_per_cust.groupby(Columns.customer_id).cumcount() + 1
    contain_counts = (
        tx.groupby([Columns.customer_id, Columns.article_id, Columns.order_date], as_index=False)
        .size()
        .rename(columns={"size": "contains_flag"})
    )
    contain_counts["contains_flag"] = 1
    contain_cum = (
        contain_counts.groupby([Columns.customer_id, Columns.article_id])["contains_flag"].cumsum().rename("contains_cum")
    )
    contain_counts["contains_cum"] = contain_cum
    # merge order_number
    contain_counts = contain_counts.merge(orders_per_cust, on=[Columns.customer_id, Columns.order_date], how="left")
    contain_counts["fraction_orders_with_article"] = safe_divide(contain_counts["contains_cum"], contain_counts["order_number"])

    # last purchase date at each row
    last_purchase = (
        tx.groupby([Columns.customer_id, Columns.article_id, Columns.order_date])["prev_date"].max().reset_index().rename(columns={"prev_date": "last_purchase_date"})
    )

    # assemble interaction features keyed by (customer_id, article_id, order_date)
    inter = (
        tx.groupby([Columns.customer_id, Columns.article_id, Columns.order_date])
        .agg(times_bought_so_far=("purchase_index", "max"), avg_gap_days=("avg_gap_days", "max"))
        .reset_index()
    )
    inter = inter.merge(last_purchase, on=[Columns.customer_id, Columns.article_id, Columns.order_date], how="left")
    inter = inter.merge(contain_counts[[Columns.customer_id, Columns.article_id, Columns.order_date, "fraction_orders_with_article"]],
                        on=[Columns.customer_id, Columns.article_id, Columns.order_date], how="left")
    return inter


def build_candidate_frame(transactions: pd.DataFrame) -> pd.DataFrame:
    # Build candidates per (customer, order_date) as all articles the customer has ever bought so far
    # This is a simplified candidate generator; in production, add popularity-based negatives.
    cust_orders = (
        transactions[[Columns.customer_id, Columns.order_date]].drop_duplicates().sort_values([Columns.customer_id, Columns.order_date])
    )
    cust_articles = (
        transactions.groupby(Columns.customer_id)[Columns.article_id].unique().rename("articles_so_far").reset_index()
    )
    # cartesian per customer across their orders
    base = cust_orders.merge(cust_articles, on=Columns.customer_id, how="left")
    exploded = base.explode("articles_so_far").rename(columns={"articles_so_far": Columns.article_id})
    # keep only articles purchased on or before the reference order_date
    first_purchase = (
        transactions.groupby([Columns.customer_id, Columns.article_id])[Columns.order_date].min().rename("first_purchase_date").reset_index()
    )
    exploded = exploded.merge(first_purchase, on=[Columns.customer_id, Columns.article_id], how="left")
    exploded = exploded[exploded["first_purchase_date"] <= exploded[Columns.order_date]]
    exploded = exploded.drop(columns=["first_purchase_date"]) 
    return exploded.reset_index(drop=True)


def assemble_features(transactions: pd.DataFrame) -> pd.DataFrame:
    # prerequisite smaller tables
    order_level = build_order_level_aggregates(transactions)
    cust_feats = compute_customer_features(order_level)
    cust_unique_articles = compute_unique_articles_so_far(transactions)
    pref_dow = compute_customer_dow_preference(order_level)
    article_feats = compute_article_features(transactions)
    inter_feats = compute_interaction_features(transactions)

    # candidate set
    candidates = build_candidate_frame(transactions)

    # enrich candidates with customer/order features
    features = candidates.merge(cust_feats[[Columns.customer_id, Columns.order_date, "order_index", "avg_days_between_orders", "avg_basket_size"]],
                                on=[Columns.customer_id, Columns.order_date], how="left")
    # merge unique article count and forward-fill per customer to emulate as-of
    features = features.merge(cust_unique_articles,
                              on=[Columns.customer_id, Columns.order_date],
                              how="left")
    features = features.sort_values([Columns.customer_id, Columns.order_date])
    features["unique_articles_so_far"] = (
        features.groupby(Columns.customer_id)["unique_articles_so_far"].ffill().fillna(0)
    )
    # direct merge for DOW pref (already at order granularity)
    features = features.merge(pref_dow, on=[Columns.customer_id, Columns.order_date], how="left")

    # interaction features as-of by (customer, article)
    inter = inter_feats.copy()
    # Robust grouped as-of merge to avoid global sort errors
    features = _asof_merge_by_multi_grouped(
        features,
        inter,
        by_cols=[Columns.customer_id, Columns.article_id],
        on=Columns.order_date,
        direction="backward",
    )

    # article features join via as-of merge per article (align last known popularity at or before order_date)
    art = article_feats.copy().sort_values([Columns.article_id, Columns.order_date])
    features = _asof_merge_by(features, art, by=Columns.article_id, on=Columns.order_date, direction="backward")

    # finalize types
    features["days_since_last_purchase"] = days_between(features["last_purchase_date"], features[Columns.order_date])

    return features


