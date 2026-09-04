"""Build leakage-safe customer features from observation-window transactions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SHARED = ROOT / "pipelines" / "shared"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))
from dataset_validation import load_table, sha256  # noqa: E402
from transaction_schema import read_transactions  # noqa: E402

LABEL_COLUMNS = ["customer_id", "cutoff_date", "observation_start", "prediction_end", "label"]
FEATURE_COLUMNS = [
    "purchase_line_count", "distinct_invoice_count", "distinct_stock_code_count", "total_quantity",
    "total_purchase_revenue", "active_purchase_days", "recency_days", "average_order_value",
    "average_items_per_invoice", "event_count", "cancellation_count", "return_count", "adjustment_count",
    "cancellation_rate", "return_rate", "event_net_revenue",
]
OUTPUT_COLUMNS = LABEL_COLUMNS + FEATURE_COLUMNS


def _read_labels(path: Path) -> pd.DataFrame:
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


def _purchase_features(purchases: pd.DataFrame, cutoff: pd.Timestamp, start: pd.Timestamp) -> pd.DataFrame:
    observed = purchases[(purchases["invoice_date"] >= start) & (purchases["invoice_date"] < cutoff)].copy()
    observed["purchase_day"] = observed["invoice_date"].dt.normalize()
    grouped = observed.groupby("customer_id", sort=False)
    features = grouped.agg(
        purchase_line_count=("invoice_id", "size"),
        distinct_invoice_count=("invoice_id", "nunique"),
        total_quantity=("quantity", "sum"),
        total_purchase_revenue=("line_revenue", "sum"),
        active_purchase_days=("purchase_day", "nunique"),
        last_purchase_date=("invoice_date", "max"),
    )
    if "stock_code" in observed.columns:
        stock_counts = observed.groupby("customer_id")["stock_code"].nunique().rename("distinct_stock_code_count")
        features = features.join(stock_counts)
    else:
        features["distinct_stock_code_count"] = 0
    features["recency_days"] = (cutoff - features.pop("last_purchase_date")).dt.total_seconds() / 86400.0
    features["average_order_value"] = features["total_purchase_revenue"] / features["distinct_invoice_count"]
    features["average_items_per_invoice"] = features["total_quantity"] / features["distinct_invoice_count"]
    return features.reset_index()


def _event_features(events: pd.DataFrame, cutoff: pd.Timestamp, start: pd.Timestamp) -> pd.DataFrame:
    observed = events[(events["invoice_date"] >= start) & (events["invoice_date"] < cutoff)].copy()
    if "event_type" not in observed.columns:
        observed["event_type"] = "purchase"
    observed["is_cancellation"] = observed["event_type"].eq("cancellation").astype("int64")
    observed["is_return"] = observed["event_type"].eq("return").astype("int64")
    observed["is_adjustment"] = observed["event_type"].eq("adjustment").astype("int64")
    features = observed.groupby("customer_id", sort=False).agg(
        event_count=("event_type", "size"),
        cancellation_count=("is_cancellation", "sum"),
        return_count=("is_return", "sum"),
        adjustment_count=("is_adjustment", "sum"),
        event_net_revenue=("line_revenue", "sum"),
    ).reset_index()
    features["cancellation_rate"] = features["cancellation_count"] / features["event_count"]
    features["return_rate"] = features["return_count"] / features["event_count"]
    return features


def build_features(events_path: Path, purchases_path: Path, labels_path: Path, output_path: Path, summary_path: Path) -> dict:
    """Join historical purchase/event aggregates to each label snapshot."""
    events = read_transactions(events_path, name="events", view="events")
    purchases = read_transactions(purchases_path, name="purchases", view="purchases")
    labels = _read_labels(labels_path)
    rows: list[pd.DataFrame] = []
    for cutoff, snapshot in labels.groupby("cutoff_date", sort=True):
        start = snapshot["observation_start"].iloc[0]
        purchase_features = _purchase_features(purchases, cutoff, start)
        event_features = _event_features(events, cutoff, start)
        enriched = snapshot.merge(purchase_features, on="customer_id", how="left", validate="one_to_one")
        enriched = enriched.merge(event_features, on="customer_id", how="left", validate="one_to_one")
        rows.append(enriched)
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=OUTPUT_COLUMNS)
    numeric_features = [column for column in FEATURE_COLUMNS if column in result.columns]
    result[numeric_features] = result[numeric_features].fillna(0)
    result = result[OUTPUT_COLUMNS].sort_values(["cutoff_date", "customer_id"]).reset_index(drop=True)
    for column in ("cutoff_date", "observation_start", "prediction_end"):
        result[column] = result[column].dt.strftime("%Y-%m-%dT%H:%M:%S")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, lineterminator="\n")
    summary = {
        "inputs": {"events": {"path": str(events_path), "sha256": sha256(events_path), "rows": int(len(events))},
                   "purchases": {"path": str(purchases_path), "sha256": sha256(purchases_path), "rows": int(len(purchases))},
                   "labels": {"path": str(labels_path), "sha256": sha256(labels_path), "rows": int(len(labels))}},
        "output": {"path": str(output_path), "sha256": sha256(output_path), "rows": int(len(result)), "columns": OUTPUT_COLUMNS},
        "cutoffs": sorted(str(value) for value in labels["cutoff_date"].unique()),
        "feature_nulls": {column: int(result[column].isna().sum()) for column in FEATURE_COLUMNS},
        "leakage_policy": "all aggregates use events and purchases with invoice_date >= observation_start and invoice_date < cutoff",
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build customer features using only each label snapshot's observation window.")
    parser.add_argument("--events", required=True, type=Path, help="All valid event rows")
    parser.add_argument("--purchases", required=True, type=Path, help="Completed purchase rows")
    parser.add_argument("--labels", required=True, type=Path, help="Repeat-purchase labels")
    parser.add_argument("--output", required=True, type=Path, help="Feature table CSV to create")
    parser.add_argument("--summary", required=True, type=Path, help="JSON audit summary")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_features(args.events, args.purchases, args.labels, args.output, args.summary)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
