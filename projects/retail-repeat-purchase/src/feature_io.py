"""File loading and output helpers for the feature pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dataset_validation import load_table, sha256
from transaction_schema import read_transactions

LABEL_COLUMNS = ["customer_id", "cutoff_date", "observation_start", "prediction_end", "label"]
FEATURE_COLUMNS = [
    "purchase_line_count", "distinct_invoice_count", "distinct_stock_code_count", "total_quantity",
    "total_purchase_revenue", "active_purchase_days", "recency_days", "average_order_value",
    "average_items_per_invoice", "event_count", "cancellation_count", "return_count", "adjustment_count",
    "cancellation_rate", "return_rate", "event_net_revenue",
]
OUTPUT_COLUMNS = LABEL_COLUMNS + FEATURE_COLUMNS


def read_labels(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Labels file does not exist: {path}")
    labels = load_table(path)
    missing = sorted(set(LABEL_COLUMNS) - set(labels.columns))
    if missing:
        raise ValueError(f"Required label columns are missing: {', '.join(missing)}")
    for column in ("cutoff_date", "observation_start", "prediction_end"):
        labels[column] = pd.to_datetime(labels[column], errors="coerce")
    if labels[["cutoff_date", "observation_start", "prediction_end"]].isna().any().any():
        raise ValueError("Labels contain unparseable window timestamps")
    labels["customer_id"] = pd.to_numeric(labels["customer_id"], errors="coerce")
    labels["label"] = pd.to_numeric(labels["label"], errors="coerce")
    if labels[["customer_id", "label"]].isna().any().any() or not labels["label"].isin([0, 1]).all():
        raise ValueError("Labels must have numeric customer_id values and binary labels")
    if labels.duplicated(["customer_id", "cutoff_date"]).any():
        raise ValueError("Labels contain duplicate customer/cutoff snapshots")
    if not (labels["observation_start"] < labels["cutoff_date"]).all() or not (labels["cutoff_date"] < labels["prediction_end"]).all():
        raise ValueError("Labels contain invalid time windows")
    labels["customer_id"] = labels["customer_id"].astype("int64")
    labels["label"] = labels["label"].astype("int64")
    return labels[LABEL_COLUMNS]


def load_feature_inputs(events_path: Path, purchases_path: Path, labels_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        read_transactions(events_path, name="events", view="events"),
        read_transactions(purchases_path, name="purchases", view="purchases"),
        read_labels(labels_path),
    )


def write_features(features: pd.DataFrame, output_path: Path) -> None:
    output = features[OUTPUT_COLUMNS].copy()
    for column in ("cutoff_date", "observation_start", "prediction_end"):
        output[column] = output[column].dt.strftime("%Y-%m-%dT%H:%M:%S")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, lineterminator="\n")


def write_summary(features: pd.DataFrame, inputs: dict[str, tuple[Path, pd.DataFrame]], output_path: Path, summary_path: Path) -> dict:
    summary = {
        "inputs": {name: {"path": str(path), "sha256": sha256(path), "rows": int(len(frame))} for name, (path, frame) in inputs.items()},
        "output": {"path": str(output_path), "sha256": sha256(output_path), "rows": int(len(features)), "columns": OUTPUT_COLUMNS},
        "cutoffs": sorted(str(value) for value in features["cutoff_date"].unique()),
        "feature_nulls": {column: int(features[column].isna().sum()) for column in FEATURE_COLUMNS},
        "leakage_policy": "all aggregates use events and purchases with invoice_date >= observation_start and invoice_date < cutoff",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return summary
