"""Apply the reviewed Online Retail transaction cleaning policy.

The raw input is never modified. The cleaned CSV and JSON summary are written
to explicit output paths supplied by the caller.
"""
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
OUTPUT_COLUMNS = ["source_row_number", "invoice_id", "stock_code", "description", "quantity", "invoice_date", "unit_price", "customer_id", "country", "line_revenue"]


def clean(input_path: Path, output_path: Path, summary_path: Path) -> dict:
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
    cancellation = frame["InvoiceNo"].str.upper().str.startswith("C", na=False) & reasons.isna()
    reasons[cancellation] = "cancellation"
    invalid_quantity = frame["Quantity"].isna() & reasons.isna()
    reasons[invalid_quantity] = "invalid_quantity"
    nonpositive_quantity = (frame["Quantity"] <= 0) & reasons.isna()
    reasons[nonpositive_quantity] = "nonpositive_quantity"
    invalid_price = frame["UnitPrice"].isna() & reasons.isna()
    reasons[invalid_price] = "invalid_unit_price"
    nonpositive_price = (frame["UnitPrice"] <= 0) & reasons.isna()
    reasons[nonpositive_price] = "nonpositive_unit_price"
    nonintegral_customer = frame["CustomerID"].notna() & (frame["CustomerID"] % 1 != 0) & reasons.isna()
    reasons[nonintegral_customer] = "nonintegral_customer_id"

    retained = reasons.isna()
    result = frame.loc[retained].copy()
    result["CustomerID"] = result["CustomerID"].astype("int64")
    result["line_revenue"] = (result["Quantity"] * result["UnitPrice"]).round(2)
    result = result.rename(columns={"InvoiceNo": "invoice_id", "StockCode": "stock_code", "Description": "description",
                                    "Quantity": "quantity", "InvoiceDate": "invoice_date", "UnitPrice": "unit_price",
                                    "CustomerID": "customer_id", "Country": "country"})[OUTPUT_COLUMNS]
    result["invoice_date"] = result["invoice_date"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, lineterminator="\n")
    exclusions = {str(key): int(value) for key, value in reasons.dropna().value_counts().sort_index().items()}
    summary = {
        "input": {"path": str(input_path), "sha256": sha256(input_path), "rows": int(len(frame)), "columns": RAW_COLUMNS},
        "output": {"path": str(output_path), "sha256": sha256(output_path), "rows": int(len(result)), "columns": OUTPUT_COLUMNS},
        "exclusions": exclusions,
        "retained": {"customers": int(result["customer_id"].nunique()), "invoices": int(result["invoice_id"].nunique()),
                     "date_min": result["invoice_date"].min() if len(result) else None,
                     "date_max": result["invoice_date"].max() if len(result) else None,
                     "line_revenue": float(result["line_revenue"].sum().round(2)) if len(result) else 0.0},
        "rules": {"require_customer_id": True, "remove_cancellations": True, "remove_nonpositive_quantity": True,
                  "remove_nonpositive_unit_price": True, "remove_exact_duplicates": True,
                  "retain_missing_description": True, "duplicate_policy": "keep_first"},
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean Online Retail transactions using an explicit, auditable policy. The raw input is never modified.")
    parser.add_argument("--input", required=True, type=Path, help="Raw CSV, Excel, Parquet, JSON, or JSONL file")
    parser.add_argument("--output", required=True, type=Path, help="Cleaned transaction CSV to create")
    parser.add_argument("--summary", required=True, type=Path, help="JSON summary of rows retained and excluded")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    summary = clean(args.input, args.output, args.summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
