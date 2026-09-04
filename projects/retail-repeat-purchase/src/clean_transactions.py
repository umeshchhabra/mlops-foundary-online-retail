"""Create auditable event and completed-purchase views from Online Retail data."""
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
from dataset_validation import SUPPORTED, load_table, sha256  # noqa: E402

RAW_COLUMNS = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"]
VIEW_COLUMNS = ["source_row_number", "invoice_id", "stock_code", "description", "quantity", "invoice_date", "unit_price", "customer_id", "country", "event_type", "line_revenue"]


def _metadata(path: Path, frame: pd.DataFrame) -> dict:
    return {"path": str(path), "sha256": sha256(path), "rows": int(len(frame)), "columns": VIEW_COLUMNS}


def clean(input_path: Path, events_path: Path, purchases_path: Path, summary_path: Path) -> dict:
    """Write valid event rows and a completed-purchase subset without editing input."""
    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    if input_path.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Unsupported input type: {input_path.suffix}")
    frame = load_table(input_path)
    missing_columns = [column for column in RAW_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Required columns are missing: {', '.join(missing_columns)}")
    frame = frame[RAW_COLUMNS].copy()
    frame.insert(0, "source_row_number", range(2, len(frame) + 2))
    frame["InvoiceNo"] = frame["InvoiceNo"].astype("string").str.strip()
    frame["StockCode"] = frame["StockCode"].astype("string").str.strip()
    frame["Country"] = frame["Country"].astype("string").str.strip()
    frame["Description"] = frame["Description"].astype("string").str.strip()
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"], errors="coerce")
    frame["Quantity"] = pd.to_numeric(frame["Quantity"], errors="coerce")
    frame["UnitPrice"] = pd.to_numeric(frame["UnitPrice"], errors="coerce")
    frame["CustomerID"] = pd.to_numeric(frame["CustomerID"], errors="coerce")

    reasons = pd.Series(pd.NA, index=frame.index, dtype="string")
    duplicate = frame.duplicated(subset=RAW_COLUMNS, keep="first")
    reasons[duplicate] = "exact_duplicate"
    missing_customer = frame["CustomerID"].isna() & reasons.isna()
    reasons[missing_customer] = "missing_customer_id"
    invalid_date = frame["InvoiceDate"].isna() & reasons.isna()
    reasons[invalid_date] = "invalid_invoice_date"
    invalid_invoice = (frame["InvoiceNo"].isna() | frame["InvoiceNo"].eq("")) & reasons.isna()
    reasons[invalid_invoice] = "invalid_invoice_id"
    invalid_stock = (frame["StockCode"].isna() | frame["StockCode"].eq("")) & reasons.isna()
    reasons[invalid_stock] = "invalid_stock_code"
    invalid_quantity = frame["Quantity"].isna() & reasons.isna()
    reasons[invalid_quantity] = "invalid_quantity"
    invalid_price = frame["UnitPrice"].isna() & reasons.isna()
    reasons[invalid_price] = "invalid_unit_price"
    nonpositive_price = (frame["UnitPrice"] <= 0) & reasons.isna()
    reasons[nonpositive_price] = "nonpositive_unit_price"
    nonintegral_customer = frame["CustomerID"].notna() & (frame["CustomerID"] % 1 != 0) & reasons.isna()
    reasons[nonintegral_customer] = "nonintegral_customer_id"

    valid = frame.loc[reasons.isna()].copy()
    valid["CustomerID"] = valid["CustomerID"].astype("int64")
    valid["event_type"] = "purchase"
    cancellation = valid["InvoiceNo"].str.upper().str.startswith("C", na=False)
    valid.loc[cancellation, "event_type"] = "cancellation"
    valid.loc[~cancellation & (valid["Quantity"] < 0), "event_type"] = "return"
    valid.loc[~cancellation & (valid["Quantity"] == 0), "event_type"] = "adjustment"
    valid["line_revenue"] = (valid["Quantity"] * valid["UnitPrice"]).round(2)
    valid = valid.rename(columns={"InvoiceNo": "invoice_id", "StockCode": "stock_code", "Description": "description",
                                  "Quantity": "quantity", "InvoiceDate": "invoice_date", "UnitPrice": "unit_price",
                                  "CustomerID": "customer_id", "Country": "country"})[VIEW_COLUMNS]
    valid["invoice_date"] = valid["invoice_date"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    purchases = valid.loc[valid["event_type"].eq("purchase")].copy()
    for path in (events_path, purchases_path, summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    valid.to_csv(events_path, index=False, lineterminator="\n")
    purchases.to_csv(purchases_path, index=False, lineterminator="\n")
    exclusions = {str(key): int(value) for key, value in reasons.dropna().value_counts().sort_index().items()}
    summary = {
        "input": {"path": str(input_path), "sha256": sha256(input_path), "rows": int(len(frame)), "columns": RAW_COLUMNS},
        "events": _metadata(events_path, valid),
        "purchases": _metadata(purchases_path, purchases),
        "exclusions": exclusions,
        "event_counts": {str(key): int(value) for key, value in valid["event_type"].value_counts().sort_index().items()},
        "rules": {"require_customer_id": True, "retain_cancellations_and_returns": True,
                  "purchases_include_only_positive_non_cancellation_rows": True,
                  "remove_nonpositive_unit_price": True, "remove_exact_duplicates": True,
                  "retain_missing_description": True, "duplicate_policy": "keep_first"},
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create auditable event and completed-purchase views. The raw input is never modified.")
    parser.add_argument("--input", required=True, type=Path, help="Raw CSV, Excel, Parquet, JSON, or JSONL file")
    parser.add_argument("--events-output", required=True, type=Path, help="All valid events, including cancellations and returns")
    parser.add_argument("--purchases-output", required=True, type=Path, help="Completed positive purchase rows for later labels/features")
    parser.add_argument("--summary", required=True, type=Path, help="JSON audit summary")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    summary = clean(args.input, args.events_output, args.purchases_output, args.summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
