"""Train deterministic local baseline classifiers from the feature table."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

METADATA_COLUMNS = {"customer_id", "cutoff_date", "observation_start", "prediction_end", "label"}


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    missing = sorted({"label"} - set(frame.columns))
    if missing:
        raise ValueError(f"Required feature columns are missing: {', '.join(missing)}")
    if {"customer_id", "cutoff_date"}.issubset(frame.columns) and frame.duplicated(["customer_id", "cutoff_date"]).any():
        raise ValueError("Feature table contains duplicate customer/cutoff rows")
    columns = [column for column in frame.columns if column not in METADATA_COLUMNS]
    if not columns:
        raise ValueError("Feature table has no model feature columns")
    return columns


def _dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    columns = _feature_columns(frame)
    features = frame[columns].apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any():
        bad = sorted(features.columns[features.isna().any()].tolist())
        raise ValueError(f"Feature table has missing or non-numeric values: {', '.join(bad)}")
    labels = pd.to_numeric(frame["label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("label must contain only binary 0/1 values")
    if labels.nunique() != 2:
        raise ValueError("Training requires both label classes")
    return features, labels.astype("int64"), columns


def _models(random_state: int) -> dict[str, object]:
    return {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=1),
    }


def _metrics(model, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(x_test)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "per_class": {
            str(key): {metric: float(value) for metric, value in values.items()}
            for key, values in report.items() if isinstance(values, dict)
        },
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
    }


def train_baselines(frame: pd.DataFrame, *, test_size: float = 0.2, random_state: int = 42) -> tuple[dict, dict]:
    """Fit baselines and return the selected artifact plus a JSON-safe report."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    features, labels, columns = _dataset(frame)
    indices = np.arange(len(frame))
    train_indices, test_indices = train_test_split(indices, test_size=test_size, stratify=labels, random_state=random_state)
    x_train, x_test = features.iloc[train_indices], features.iloc[test_indices]
    y_train, y_test = labels.iloc[train_indices], labels.iloc[test_indices]
    fitted = {}
    results = {}
    for name, model in _models(random_state).items():
        model.fit(x_train, y_train)
        fitted[name] = model
        results[name] = _metrics(model, x_test, y_test)
    priority = {"dummy": 0, "logistic_regression": 1, "random_forest": 2}
    selected = max(results, key=lambda name: (results[name]["macro_f1"], results[name]["accuracy"], -priority[name]))
    artifact = {
        "model": fitted[selected],
        "feature_columns": columns,
        "target_column": "label",
        "selected_model": selected,
        "random_state": random_state,
        "test_size": test_size,
        "dependencies": {"numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "joblib": joblib.__version__},
    }
    report = {
        "rows": int(len(frame)),
        "feature_columns": columns,
        "train_rows": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
        "random_state": random_state,
        "test_size": test_size,
        "selected_model": selected,
        "models": results,
        "dependencies": {"numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "joblib": joblib.__version__},
    }
    return artifact, report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and compare local baseline classifiers from a feature CSV.")
    parser.add_argument("--features", required=True, type=Path, help="Feature CSV containing label and numeric features")
    parser.add_argument("--model", required=True, type=Path, help="Joblib artifact to create")
    parser.add_argument("--report", required=True, type=Path, help="JSON evaluation report to create")
    parser.add_argument("--test-size", type=float, default=0.2, help="Stratified test fraction (default: 0.2)")
    parser.add_argument("--random-state", type=int, default=42, help="Split and model seed (default: 42)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.features.exists() or not args.features.is_file():
        raise FileNotFoundError(f"Features file does not exist: {args.features}")
    frame = pd.read_csv(args.features)
    artifact, report = train_baselines(frame, test_size=args.test_size, random_state=args.random_state)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.model)
    report["input_sha256"] = _sha256(args.features)
    report["model"] = {"path": str(args.model), "sha256": _sha256(args.model)}
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
