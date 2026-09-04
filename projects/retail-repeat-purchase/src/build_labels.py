"""Build leakage-safe repeat-purchase labels from completed purchases."""
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

REQUIRED_COLUMNS = {"customer_id", "invoice_id", "invoice_date", "quantity", "line_revenue"}
LABEL_COLUMNS = ["customer_id", "cutoff_date", "observation_start", "prediction_end", "label"]


def build_labels(
    purchases_path: Path,
    output_path: Path,
    summary_path: Path,
    cutoffs: list[pd.Timestamp],
    observation_days: int = 90,
    prediction_days: int = 30,
) -> dict:
    """Create one row per eligible customer and cutoff using half-open windows."""
    if not purchases_path.exists() or not purchases_path.is_file():
        raise FileNotFoundError(f"Purchases file does not exist: {purchases_path}")
    if purchases_path.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Unsupported input type: {purchases_path.suffix}")
    if observation_days <= 0 or prediction_days <= 0:
        raise ValueError("observation_days and prediction_days must be positive")
    if not cutoffs:
        raise ValueError("At least one cutoff is required")
    cutoffs = sorted(set(pd.Timestamp(cutoff) for cutoff in cutoffs))

    frame = load_table(purchases_path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Required purchase columns are missing: {', '.join(missing)}")
    if "event_type" in frame.columns and not frame["event_type"].eq("purchase").all():
        raise ValueError("Label input must contain only event_type=purchase rows")
    frame["invoice_date"] = pd.to_datetime(frame["invoice_date"], errors="coerce")
    if frame["invoice_date"].isna().any():
        raise ValueError("Purchase input contains unparseable invoice_date values")
    if frame["customer_id"].isna().any():
        raise ValueError("Purchase input contains missing customer_id values")
    if (pd.to_numeric(frame["quantity"], errors="coerce") <= 0).any():
        raise ValueError("Purchase input contains non-positive quantity values")
    frame["customer_id"] = pd.to_numeric(frame["customer_id"], errors="raise").astype("int64")
    frame["quantity"] = pd.to_numeric(frame["quantity"], errors="raise")
    frame["line_revenue"] = pd.to_numeric(frame["line_revenue"], errors="raise")
    min_date = frame["invoice_date"].min()
    max_date = frame["invoice_date"].max()

    rows: list[dict] = []
    per_cutoff: list[dict] = []
    for cutoff in cutoffs:
        observation_start = cutoff - pd.to_timedelta(observation_days, unit="D")
        prediction_end = cutoff + pd.to_timedelta(prediction_days, unit="D")
        if observation_start < min_date:
            raise ValueError(f"Cutoff {cutoff.isoformat()} starts before the available purchase history")
        if prediction_end > max_date:
            raise ValueError(f"Cutoff {cutoff.isoformat()} ends after the available purchase history")
        observed = frame[(frame["invoice_date"] >= observation_start) & (frame["invoice_date"] < cutoff)]
        future = frame[(frame["invoice_date"] >= cutoff) & (frame["invoice_date"] < prediction_end)]
        observed_customers = observed["customer_id"].drop_duplicates().sort_values()
        future_customers = set(future["customer_id"].drop_duplicates())
        positives = 0
        for customer_id in observed_customers:
            label = int(customer_id in future_customers)
            positives += label
            rows.append({
                "customer_id": int(customer_id),
                "cutoff_date": cutoff.strftime("%Y-%m-%dT%H:%M:%S"),
                "observation_start": observation_start.strftime("%Y-%m-%dT%H:%M:%S"),
                "prediction_end": prediction_end.strftime("%Y-%m-%dT%H:%M:%S"),
                "label": label,
            })
        per_cutoff.append({"cutoff_date": cutoff.strftime("%Y-%m-%dT%H:%M:%S"), "eligible_customers": int(len(observed_customers)), "positive_labels": positives, "negative_labels": int(len(observed_customers) - positives)})

    labels = pd.DataFrame(rows, columns=LABEL_COLUMNS)
    if not labels.empty:
        labels = labels.sort_values(["cutoff_date", "customer_id"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(output_path, index=False, lineterminator="\n")
    summary = {
        "input": {"path": str(purchases_path), "sha256": sha256(purchases_path), "rows": int(len(frame))},
        "output": {"path": str(output_path), "sha256": sha256(output_path), "rows": int(len(labels)), "columns": LABEL_COLUMNS},
        "observation_days": observation_days,
        "prediction_days": prediction_days,
        "cutoffs": per_cutoff,
        "label_counts": {str(key): int(value) for key, value in labels["label"].value_counts().sort_index().items()} if not labels.empty else {},
        "window_policy": "observation is [cutoff-observation_days, cutoff); prediction is [cutoff, cutoff+prediction_days)",
        "eligibility_policy": "customer must have at least one completed purchase in the observation window",
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build repeat-purchase labels without using future data in the observation window.")
    parser.add_argument("--purchases", required=True, type=Path, help="Completed-purchase CSV")
    parser.add_argument("--output", required=True, type=Path, help="Label CSV to create")
    parser.add_argument("--summary", required=True, type=Path, help="JSON audit summary")
    parser.add_argument("--cutoff", action="append", required=True, help="Cutoff timestamp; repeat for multiple snapshots")
    parser.add_argument("--observation-days", type=int, default=90, help="History window before each cutoff (default: 90)")
    parser.add_argument("--prediction-days", type=int, default=30, help="Future label window after each cutoff (default: 30)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cutoffs = [pd.Timestamp(value) for value in args.cutoff]
    summary = build_labels(args.purchases, args.output, args.summary, cutoffs, args.observation_days, args.prediction_days)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
