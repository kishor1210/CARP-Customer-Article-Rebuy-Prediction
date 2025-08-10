from __future__ import annotations

import os
from pathlib import Path
import sys

import click
import pandas as pd

# Ensure local src/ is importable when running as a script
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.carp.config import Columns, Paths
from src.carp.data_loader import read_transactions_csv
from src.carp.feature_engineering import assemble_features
from src.carp.labeling import build_labels
from src.carp.model import FEATURE_COLUMNS_DEFAULT, evaluate_auc, prepare_train_valid, train_xgb
from src.carp.serve import generate_recommendations
from src.carp.utils import save_json


def ensure_parents(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--input", "input_path", default=Paths.default_input, show_default=True)
@click.option("--output", "output_path", default=Paths.default_dataset, show_default=True)
def build_dataset(input_path: str, output_path: str):
    """Build feature table and labels, write parquet."""
    tx = read_transactions_csv(input_path)
    features = assemble_features(tx)
    labels = build_labels(tx)
    ds = features.merge(labels, on=[Columns.customer_id, Columns.order_date, Columns.article_id], how="left")
    ds["label"] = ds["label"].fillna(0).astype(int)
    ensure_parents(output_path)
    ds.to_parquet(output_path, index=False)
    click.echo(f"Wrote dataset: {output_path}, rows={len(ds):,}")


@cli.command()
@click.option("--dataset", "dataset_path", default=Paths.default_dataset, show_default=True)
@click.option("--train-end", "train_end_date", required=True, help="YYYY-MM-DD")
@click.option("--model-out", default=Paths.default_model, show_default=True)
@click.option("--features-out", default=Paths.default_feature_cols, show_default=True)
def train(dataset_path: str, train_end_date: str, model_out: str, features_out: str):
    ds = pd.read_parquet(dataset_path)
    train_d, valid_d, feature_cols = prepare_train_valid(ds, train_end_date, FEATURE_COLUMNS_DEFAULT)
    bst = train_xgb(train_d, valid_d)
    auc = evaluate_auc(bst, valid_d)
    ensure_parents(model_out)
    bst.save_model(model_out)
    ensure_parents(features_out)
    save_json(features_out, feature_cols)
    click.echo(f"Model saved: {model_out} | AUC(valid)={auc:.4f}")


@cli.command()
@click.option("--input", "input_path", default=Paths.default_input, show_default=True)
@click.option("--model", "model_path", default=Paths.default_model, show_default=True)
@click.option("--features", "features_path", default=Paths.default_feature_cols, show_default=True)
@click.option("--topn", default=5, show_default=True)
@click.option("--output", "output_path", default=Paths.default_predictions, show_default=True)
def predict(input_path: str, model_path: str, features_path: str, topn: int, output_path: str):
    tx = read_transactions_csv(input_path)
    recs = generate_recommendations(tx, model_path, features_path, top_n=topn)
    ensure_parents(output_path)
    recs.to_parquet(output_path, index=False)
    click.echo(f"Wrote recommendations: {output_path}, rows={len(recs):,}")


if __name__ == "__main__":
    cli()


