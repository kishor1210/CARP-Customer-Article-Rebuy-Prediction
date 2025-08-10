from __future__ import annotations

import pandas as pd

from .config import Columns


def build_labels(transactions: pd.DataFrame) -> pd.DataFrame:
    # For each (customer, order_date), find the next order's article set
    orders = (
        transactions[[Columns.customer_id, Columns.order_date]].drop_duplicates().sort_values([Columns.customer_id, Columns.order_date])
    )
    orders["next_order_date"] = orders.groupby(Columns.customer_id)[Columns.order_date].shift(-1)

    # article presence in next order
    next_order_articles = (
        transactions.merge(orders[[Columns.customer_id, Columns.order_date, "next_order_date"]],
                           on=[Columns.customer_id, Columns.order_date], how="left")
    )
    mask = next_order_articles[Columns.order_date] == next_order_articles["next_order_date"]
    # rows where order_date equals next_order_date give us next order's item rows per customer. Build sets per (customer, prev_order_date).
    next_articles = (
        transactions[[Columns.customer_id, Columns.order_date, Columns.article_id]]
        .rename(columns={Columns.order_date: "actual_order_date"})
        .merge(orders.rename(columns={Columns.order_date: "prev_order_date"}), left_on=[Columns.customer_id, "actual_order_date"], right_on=[Columns.customer_id, "next_order_date"], how="inner")
        [[Columns.customer_id, "prev_order_date", Columns.article_id]]
        .rename(columns={"prev_order_date": Columns.order_date})
    )
    next_articles["label"] = 1
    labels = next_articles
    return labels


