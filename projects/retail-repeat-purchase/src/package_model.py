"""Validate and package a tuned model with its input contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema(frame: pd.DataFrame, columns: list[str]) -> dict:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Feature input is missing columns: {', '.join(missing)}")
    schema = {}
    for column in columns:
        series = frame[column]
        if not is_numeric_dtype(series) or series.isna().any():
            raise ValueError(f"Feature input column {column} must be numeric and non-null")
        schema[column] = {"dtype": str(series.dtype), "kind": "numeric", "nullable": False}
    return schema


def _prediction_check(model, frame: pd.DataFrame, columns: list[str]) -> dict:
    predictions = np.asarray(model.predict(frame[columns]))
    repeated = np.asarray(model.predict(frame[columns]))
    if not np.array_equal(predictions, repeated):
        raise ValueError("Model predictions are not deterministic")
    encoded = json.dumps(predictions.tolist(), separators=(",", ":")).encode("utf-8")
    return {
        "rows": int(len(predictions)),
        "positive_predictions": int(np.sum(predictions == 1)),
        "prediction_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def package_model(
    model_path: Path,
    report_path: Path,
    features_path: Path,
    code_path: Path,
    output_path: Path,
    manifest_path: Path,
) -> dict:
    """Require baseline improvement, attach schema and lineage, and verify predictions."""
    artifact = joblib.load(model_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict) or not callable(getattr(artifact.get("model"), "predict", None)):
        raise ValueError("Model artifact must contain a fitted model with predict")
    selected = artifact.get("selected_model")
    if selected != report.get("selected_model"):
        raise ValueError("Model and report selected_model values do not match")
    baseline_f1 = float(report["baseline"]["test"]["macro_f1"])
    selected_f1 = float(report["models"][selected]["test"]["macro_f1"])
    if not selected_f1 > baseline_f1:
        raise ValueError("Selected model does not improve test macro-F1 over the dummy baseline")
    frame = pd.read_csv(features_path)
    columns = list(artifact.get("feature_columns", []))
    if not columns:
        raise ValueError("Model artifact has no feature_columns")
    schema = _schema(frame, columns)
    input_hash = sha256(features_path)
    if report.get("input_sha256") != input_hash:
        raise ValueError("Feature input hash does not match the tuning report")
    prediction_check = _prediction_check(artifact["model"], frame, columns)
    package = dict(artifact)
    package.update({
        "input_schema": schema,
        "dataset_sha256": input_hash,
        "report_sha256": sha256(report_path),
        "training_code_sha256": sha256(code_path),
        "evaluation": {
            "baseline_macro_f1": baseline_f1,
            "selected_test_macro_f1": selected_f1,
            "selected_test_accuracy": float(report["models"][selected]["test"]["accuracy"]),
        },
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, output_path)
    manifest = {
        "model": {"path": str(output_path), "sha256": sha256(output_path)},
        "selected_model": selected,
        "feature_columns": columns,
        "input_schema": schema,
        "dataset_sha256": input_hash,
        "report_sha256": package["report_sha256"],
        "training_code_sha256": package["training_code_sha256"],
        "evaluation": package["evaluation"],
        "prediction_check": prediction_check,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and package a tuned model with schema and lineage metadata.")
    parser.add_argument("--model", required=True, type=Path, help="Tuned Joblib artifact")
    parser.add_argument("--report", required=True, type=Path, help="Tuning report JSON")
    parser.add_argument("--features", required=True, type=Path, help="Feature CSV consumed during tuning")
    parser.add_argument("--code", required=True, type=Path, help="Training script to hash")
    parser.add_argument("--output", required=True, type=Path, help="Packaged Joblib artifact")
    parser.add_argument("--manifest", required=True, type=Path, help="Package manifest JSON")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    for path in (args.model, args.report, args.features, args.code):
        if not path.is_file():
            raise FileNotFoundError(f"Required file does not exist: {path}")
    manifest = package_model(args.model, args.report, args.features, args.code, args.output, args.manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
