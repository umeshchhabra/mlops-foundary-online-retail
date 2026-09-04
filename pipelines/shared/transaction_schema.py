"""Schema normalization and view-specific validation for cleaned transactions."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from dataset_validation import SUPPORTED, load_table

BASE_COLUMNS = (
    "customer_id",
    "invoice_id",
    "stock_code",
    "invoice_date",
    "quantity",
    "line_revenue",
    "event_type",
)
ALLOWED_EVENT_TYPES = frozenset({"purchase", "cancellation", "return", "adjustment"})


class DataValidationError(ValueError):
    """Raised when a cleaned transaction view violates its data contract."""


def _blank_values(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype("string").str.strip().eq("")


def _normalize(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    normalized = frame.copy()
    errors: list[str] = []

    for column in ("invoice_id", "stock_code", "event_type"):
        normalized[column] = normalized[column].astype("string").str.strip()
        if _blank_values(normalized[column]).any():
            errors.append(f"{column}: {int(_blank_values(normalized[column]).sum())} missing")

    numeric_columns = ("customer_id", "quantity", "line_revenue")
    for column in numeric_columns:
        raw = normalized[column]
        missing = _blank_values(raw)
        parsed = pd.to_numeric(raw, errors="coerce")
        invalid = (~missing) & parsed.isna()
        if missing.any():
            errors.append(f"{column}: {int(missing.sum())} missing")
        if invalid.any():
            errors.append(f"{column}: {int(invalid.sum())} unparseable")
        normalized[column] = parsed

    raw_dates = normalized["invoice_date"]
    missing_dates = _blank_values(raw_dates)
    parsed_dates = pd.to_datetime(raw_dates, errors="coerce")
    invalid_dates = (~missing_dates) & parsed_dates.isna()
    if missing_dates.any():
        errors.append(f"invoice_date: {int(missing_dates.sum())} missing")
    if invalid_dates.any():
        errors.append(f"invoice_date: {int(invalid_dates.sum())} unparseable")
    normalized["invoice_date"] = parsed_dates

    unknown_events = ~normalized["event_type"].isin(ALLOWED_EVENT_TYPES)
    if unknown_events.any():
        invalid_types = sorted(set(normalized.loc[unknown_events, "event_type"]))
        errors.append(f"event_type: unknown values: {', '.join(invalid_types)}")

    if errors:
        raise DataValidationError(f"{name} validation failed: " + "; ".join(errors))
    if (normalized["customer_id"] % 1 != 0).any():
        count = int((normalized["customer_id"] % 1 != 0).sum())
        raise DataValidationError(f"{name} validation failed: customer_id: {count} non-integral values")
    normalized["customer_id"] = normalized["customer_id"].astype("int64")
    return normalized


def read_transactions(path: Path, *, name: str, view: str) -> pd.DataFrame:
    """Load and validate a cleaned events or purchases view."""
    if view not in {"events", "purchases"}:
        raise ValueError(f"Unknown transaction view: {view}")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{name} file does not exist: {path}")
    if path.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Unsupported {name} type: {path.suffix}")
    frame = load_table(path)
    missing = sorted(set(BASE_COLUMNS) - set(frame.columns))
    if missing:
        raise DataValidationError(f"{name} is missing columns: {', '.join(missing)}")
    normalized = _normalize(frame, name)
    if view == "purchases":
        nonpositive = normalized["quantity"] <= 0
        not_purchase = normalized["event_type"] != "purchase"
        if nonpositive.any():
            raise DataValidationError(f"{name} validation failed: quantity: {int(nonpositive.sum())} non-positive values")
        if not_purchase.any():
            raise DataValidationError(f"{name} validation failed: event_type must be purchase for all rows")
    return normalized
