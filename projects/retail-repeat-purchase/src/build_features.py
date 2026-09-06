"""Calculate customer features from historical transaction DataFrames."""
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

from feature_io import FEATURE_COLUMNS, LABEL_COLUMNS, load_feature_inputs, write_features, write_summary  # noqa: E402

OUTPUT_COLUMNS = LABEL_COLUMNS + FEATURE_COLUMNS


def _purchase_features(purchases: pd.DataFrame, cutoff: pd.Timestamp, start: pd.Timestamp) -> pd.DataFrame:
    observed = purchases[purchases["invoice_date"].between(start, cutoff, inclusive="left")].copy()
    observed["purchase_day"] = observed["invoice_date"].dt.normalize()
    features = observed.groupby("customer_id", sort=False).agg(
        purchase_line_count=("invoice_id", "size"),
        distinct_invoice_count=("invoice_id", "nunique"),
        distinct_stock_code_count=("stock_code", "nunique"),
        total_quantity=("quantity", "sum"),
        total_purchase_revenue=("line_revenue", "sum"),
        active_purchase_days=("purchase_day", "nunique"),
        last_purchase_date=("invoice_date", "max"),
    )
    features["recency_days"] = (cutoff - features.pop("last_purchase_date")).dt.total_seconds() / 86400.0
    features["average_order_value"] = features["total_purchase_revenue"] / features["distinct_invoice_count"]
    features["average_items_per_invoice"] = features["total_quantity"] / features["distinct_invoice_count"]
    return features.reset_index()


def _event_features(events: pd.DataFrame, cutoff: pd.Timestamp, start: pd.Timestamp) -> pd.DataFrame:
    observed = events[events["invoice_date"].between(start, cutoff, inclusive="left")].copy()
    indicators = pd.get_dummies(observed["event_type"], dtype="int64")
    for event_type in ("cancellation", "return", "adjustment"):
        if event_type not in indicators:
            indicators[event_type] = 0
    observed = observed.join(indicators[["cancellation", "return", "adjustment"]].rename(columns={
        "cancellation": "cancellation_count", "return": "return_count", "adjustment": "adjustment_count",
    }))
    features = observed.groupby("customer_id", sort=False).agg(
        event_count=("event_type", "size"),
        cancellation_count=("cancellation_count", "sum"),
        return_count=("return_count", "sum"),
        adjustment_count=("adjustment_count", "sum"),
        event_net_revenue=("line_revenue", "sum"),
    ).reset_index()
    features["cancellation_rate"] = features["cancellation_count"] / features["event_count"]
    features["return_rate"] = features["return_count"] / features["event_count"]
    return features


def _snapshot_features(events: pd.DataFrame, purchases: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
    cutoff = snapshot["cutoff_date"].iloc[0]
    start = snapshot["observation_start"].iloc[0]
    result = snapshot.merge(_purchase_features(purchases, cutoff, start), on="customer_id", how="left", validate="one_to_one")
    return result.merge(_event_features(events, cutoff, start), on="customer_id", how="left", validate="one_to_one")


def build_features(events: pd.DataFrame, purchases: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Return one feature row per label snapshot using historical rows only."""
    snapshots = [_snapshot_features(events, purchases, snapshot) for _, snapshot in labels.groupby("cutoff_date", sort=True)]
    result = pd.concat(snapshots, ignore_index=True) if snapshots else pd.DataFrame(columns=OUTPUT_COLUMNS)
    result[FEATURE_COLUMNS] = result[FEATURE_COLUMNS].fillna(0)
    return result[OUTPUT_COLUMNS].sort_values(["cutoff_date", "customer_id"]).reset_index(drop=True)


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
    events, purchases, labels = load_feature_inputs(args.events, args.purchases, args.labels)
    features = build_features(events, purchases, labels)
    write_features(features, args.output)
    summary = write_summary(
        features,
        {"events": (args.events, events), "purchases": (args.purchases, purchases), "labels": (args.labels, labels)},
        args.output,
        args.summary,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
