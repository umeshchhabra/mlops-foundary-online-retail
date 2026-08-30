"""Inspect a tabular dataset and return a machine-readable validation report.

The validator is intentionally read-only: it never modifies the input file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

SUPPORTED = {".csv", ".xlsx", ".xls", ".parquet", ".json", ".jsonl"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_table(path: Path, sheet: str | int | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported file type '{suffix}'. Supported: {', '.join(sorted(SUPPORTED))}")


def json_value(value):
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return str(value)


def validate(path: Path, *, required_columns=None, date_columns=None,
             expected_columns=None, sample_rows=5, max_size_bytes=None,
             expected_sha256=None) -> dict:
    report = {
        "valid": False,
        "file": {"path": str(path), "name": path.name},
        "integrity": {},
        "table": {},
        "schema": {},
        "quality": {},
        "errors": [],
        "warnings": [],
    }
    if not path.exists():
        report["errors"].append(f"Input file does not exist: {path}")
        return report
    if not path.is_file():
        report["errors"].append(f"Input path is not a file: {path}")
        return report
    if path.suffix.lower() not in SUPPORTED:
        report["errors"].append(f"Unsupported file type: {path.suffix or '(none)'}")
        return report
    size = path.stat().st_size
    report["integrity"] = {"size_bytes": size, "sha256": sha256(path), "readable": True}
    if expected_sha256 is not None:
        if report["integrity"]["sha256"].lower() != expected_sha256.lower():
            report["errors"].append("SHA-256 does not match --expected-sha256")
    if max_size_bytes is not None and size > max_size_bytes:
        report["errors"].append(f"File size {size} exceeds limit {max_size_bytes}")
    try:
        frame = load_table(path)
    except Exception as exc:
        report["integrity"]["readable"] = False
        report["errors"].append(f"Could not parse tabular data: {type(exc).__name__}: {exc}")
        return report
    columns = [str(column) for column in frame.columns]
    report["table"] = {"rows": int(len(frame)), "columns": int(len(columns)), "column_names": columns}
    report["schema"] = {"dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
                         "duplicate_column_names": len(columns) != len(set(columns))}
    report["quality"] = {
        "missing_by_column": {column: int(count) for column, count in frame.isna().sum().items() if count},
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "numeric_non_finite_by_column": {},
        "sample": [{str(k): json_value(v) for k, v in row.items()} for row in frame.head(sample_rows).to_dict(orient="records")],
    }
    for column in frame.select_dtypes(include="number").columns:
        count = int((~frame[column].map(lambda value: math.isfinite(float(value)) if pd.notna(value) else True)).sum())
        if count:
            report["quality"]["numeric_non_finite_by_column"][str(column)] = count
    if report["schema"]["duplicate_column_names"]:
        report["errors"].append("Column names are not unique")
    if not columns:
        report["errors"].append("Dataset has no columns")
    if len(frame) == 0:
        report["errors"].append("Dataset has no rows")
    for column in required_columns or []:
        if column not in columns:
            report["errors"].append(f"Required column is missing: {column}")
    if expected_columns is not None and columns != expected_columns:
        report["errors"].append("Column names/order do not match --expected-columns")
        report["schema"]["expected_columns"] = expected_columns
    for column in date_columns or []:
        if column not in frame:
            report["errors"].append(f"Date column is missing: {column}")
            continue
        parsed = pd.to_datetime(frame[column], errors="coerce")
        invalid = int(parsed.isna().sum())
        report["schema"].setdefault("date_columns", {})[column] = {
            "invalid_values": invalid,
            "min": json_value(parsed.min()) if invalid < len(parsed) else None,
            "max": json_value(parsed.max()) if invalid < len(parsed) else None,
        }
        if invalid:
            report["errors"].append(f"Date column '{column}' has {invalid} unparseable values")
    if report["quality"]["missing_cells"]:
        report["warnings"].append("Dataset contains missing values")
    if report["quality"]["duplicate_rows"]:
        report["warnings"].append("Dataset contains duplicate rows")
    if report["quality"]["numeric_non_finite_by_column"]:
        report["warnings"].append("Dataset contains non-finite numeric values")
    report["valid"] = not report["errors"]
    return report


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        prog="validate_dataset.py",
        description="Read a tabular dataset and report integrity, schema, dimensions, and quality issues. The input is never modified.",
        epilog="Example: python pipelines/shared/dataset_validation.py --input 'projects/retail-repeat-purchase/data/raw/Online Retail.xlsx' --date-column InvoiceDate --required-column CustomerID",
    )
    cli.add_argument("--input", required=True, type=Path, help="CSV, Excel, Parquet, JSON, or JSONL file to inspect")
    cli.add_argument("--report", type=Path, help="Optional path for a JSON report; stdout always receives the report")
    cli.add_argument("--required-column", action="append", default=[], help="Column that must exist; repeat for multiple columns")
    cli.add_argument("--date-column", action="append", default=[], help="Column to parse and summarize as dates; repeatable")
    cli.add_argument("--expected-columns", help="Comma-separated column names in the required order")
    cli.add_argument("--sample-rows", type=int, default=5, help="Number of sample rows in the report (default: 5)")
    cli.add_argument("--max-size-mb", type=float, help="Fail if the file exceeds this size")
    cli.add_argument("--expected-sha256", help="Expected SHA-256 digest; fail if the file differs")
    cli.add_argument("--fail-on-warning", action="store_true", help="Exit with status 1 when warnings are found")
    return cli


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.sample_rows < 0:
        parser().error("--sample-rows must be non-negative")
    expected = args.expected_columns.split(",") if args.expected_columns is not None else None
    report = validate(args.input, required_columns=args.required_column, date_columns=args.date_column,
                      expected_columns=expected, sample_rows=args.sample_rows,
                      max_size_bytes=int(args.max_size_mb * 1024 * 1024) if args.max_size_mb is not None else None,
                      expected_sha256=args.expected_sha256)
    rendered = json.dumps(report, indent=2, sort_keys=True, default=json_value)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["valid"] and not (args.fail_on_warning and report["warnings"]) else 1


if __name__ == "__main__":
    sys.exit(main())
