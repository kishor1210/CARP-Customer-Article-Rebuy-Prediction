from __future__ import annotations

from pathlib import Path
import sys
import io

import pandas as pd
import streamlit as st

# Make local src importable
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


st.set_page_config(page_title="CARP: Customer Article Rebuy Prediction", layout="wide")
st.title("CARP: Customer Article Rebuy Prediction")

with st.sidebar:
    st.header("Actions")
    action = st.radio("Choose an action", ["Build dataset", "Train model", "Predict"], index=0)

st.write("Use the sidebar to select an action.")


def upload_or_path(default_path: str, key: str) -> str:
    st.markdown("#### Input Transactions")
    uploaded = st.file_uploader("Upload transactions CSV", type=["csv"], key=key)
    if uploaded is not None:
        # Save uploaded file to a temp path
        data = uploaded.getvalue()
        tmp_path = Path("data/raw/_uploaded_transactions.csv")
        ensure_parents(str(tmp_path))
        with open(tmp_path, "wb") as f:
            f.write(data)
        return str(tmp_path)
    else:
        st.caption(f"Or use default: {default_path}")
        return default_path


if action == "Build dataset":
    input_path = upload_or_path(Paths.default_input, key="build_input")
    output_path = st.text_input("Output dataset path (parquet)", value=Paths.default_dataset)
    if st.button("Build dataset", type="primary"):
        try:
            tx = read_transactions_csv(input_path)
            features = assemble_features(tx)
            labels = build_labels(tx)
            ds = features.merge(labels, on=[Columns.customer_id, Columns.order_date, Columns.article_id], how="left")
            ds["label"] = ds["label"].fillna(0).astype(int)
            ensure_parents(output_path)
            ds.to_parquet(output_path, index=False)
            st.success(f"Wrote dataset: {output_path} | rows={len(ds):,}")
            st.dataframe(ds.head(50))
        except Exception as e:
            st.error(f"Failed to build dataset: {e}")

elif action == "Train model":
    dataset_path = st.text_input("Dataset path (parquet)", value=Paths.default_dataset)
    train_end_date = st.text_input("Train end date (YYYY-MM-DD)", value="2024-05-31")
    model_out = st.text_input("Model output path", value=Paths.default_model)
    features_out = st.text_input("Feature columns output path", value=Paths.default_feature_cols)
    if st.button("Train model", type="primary"):
        try:
            ds = pd.read_parquet(dataset_path)
            train_d, valid_d, feature_cols = prepare_train_valid(ds, train_end_date, FEATURE_COLUMNS_DEFAULT)
            bst = train_xgb(train_d, valid_d)
            auc = evaluate_auc(bst, valid_d)
            ensure_parents(model_out)
            bst.save_model(model_out)
            ensure_parents(features_out)
            save_json(features_out, feature_cols)
            st.success(f"Model saved: {model_out} | AUC(valid)={auc:.4f}")
        except Exception as e:
            st.error(f"Failed to train: {e}")

elif action == "Predict":
    input_path = upload_or_path(Paths.default_input, key="predict_input")
    model_path = st.text_input("Model path", value=Paths.default_model)
    features_path = st.text_input("Feature columns path", value=Paths.default_feature_cols)
    topn = st.number_input("Top-N recommendations per customer", value=5, min_value=1, max_value=50, step=1)
    output_path = st.text_input("Output recommendations path (parquet)", value=Paths.default_predictions)
    if st.button("Predict", type="primary"):
        try:
            tx = read_transactions_csv(input_path)
            recs = generate_recommendations(tx, model_path, features_path, top_n=int(topn))
            ensure_parents(output_path)
            recs.to_parquet(output_path, index=False)
            st.success(f"Wrote recommendations: {output_path} | rows={len(recs):,}")
            st.dataframe(recs.head(100))
        except Exception as e:
            st.error(f"Failed to predict: {e}")


