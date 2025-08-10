from dataclasses import dataclass


@dataclass(frozen=True)
class Columns:
    customer_id: str = "customer_id"
    article_id: str = "article_id"
    order_date: str = "order_date"
    quantity: str = "quantity"


@dataclass(frozen=True)
class Paths:
    default_input: str = "data/raw/transactions.csv"
    default_dataset: str = "data/processed/carp_dataset.parquet"
    default_model: str = "models/carp_xgb.json"
    default_feature_cols: str = "models/feature_columns.json"
    default_predictions: str = "data/predictions/recommendations.parquet"


