"""Tune the repeat-purchase baselines without touching the held-out test set."""
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
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_baselines import _dataset, _metrics


SEARCHES = {
    "logistic_regression": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]),
        {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__class_weight": [None, "balanced"],
        },
    ),
    "random_forest": (
        RandomForestClassifier(n_jobs=1),
        {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_leaf": [1, 2],
        },
    ),
}


def _split(frame: pd.DataFrame, test_size: float, random_state: int):
    features, labels, columns = _dataset(frame)
    indices = np.arange(len(frame))
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, stratify=labels, random_state=random_state
    )
    return (
        features.iloc[train_indices], features.iloc[test_indices],
        labels.iloc[train_indices], labels.iloc[test_indices], columns,
    )


def tune_models(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 3,
    max_configs: int = 20,
) -> tuple[dict, dict]:
    """Tune two models on training rows and evaluate the winner once on test rows."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2")
    x_train, x_test, y_train, y_test, columns = _split(frame, test_size, random_state)
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(x_train, y_train)
    baseline_metrics = _metrics(dummy, x_test, y_test)
    fitted = {}
    results = {}
    priority = {"logistic_regression": 0, "random_forest": 1}
    for name, (estimator, grid) in SEARCHES.items():
        configurations = len(list(ParameterGrid(grid)))
        if configurations > max_configs:
            raise ValueError(f"{name} has {configurations} configurations; limit is {max_configs}")
        if name == "logistic_regression":
            estimator.set_params(classifier__random_state=random_state)
        else:
            estimator.set_params(random_state=random_state)
        search = GridSearchCV(
            estimator,
            grid,
            scoring={"macro_f1": "f1_macro", "accuracy": "accuracy"},
            refit="macro_f1",
            cv=splitter,
            n_jobs=1,
            return_train_score=False,
        )
        search.fit(x_train, y_train)
        index = search.best_index_
        cv_f1 = float(search.cv_results_["mean_test_macro_f1"][index])
        cv_accuracy = float(search.cv_results_["mean_test_accuracy"][index])
        results[name] = {
            "configurations": configurations,
            "best_params": search.best_params_,
            "cv_macro_f1": cv_f1,
            "cv_macro_f1_std": float(search.cv_results_["std_test_macro_f1"][index]),
            "cv_accuracy": cv_accuracy,
            "test": _metrics(search.best_estimator_, x_test, y_test),
        }
        fitted[name] = search.best_estimator_

    selected = max(
        results,
        key=lambda name: (results[name]["cv_macro_f1"], results[name]["cv_accuracy"], -priority[name]),
    )
    dependencies = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    artifact = {
        "model": fitted[selected],
        "feature_columns": columns,
        "target_column": "label",
        "selected_model": selected,
        "selected_params": results[selected]["best_params"],
        "random_state": random_state,
        "test_size": test_size,
        "cv_folds": cv_folds,
        "scoring": "f1_macro",
        "dependencies": dependencies,
    }
    report = {
        "rows": int(len(frame)),
        "feature_columns": columns,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "random_state": random_state,
        "test_size": test_size,
        "cv_folds": cv_folds,
        "scoring": "f1_macro",
        "selected_model": selected,
        "baseline": {"model": "dummy", "test": baseline_metrics},
        "models": results,
        "dependencies": dependencies,
    }
    return artifact, report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune repeat-purchase classifiers with bounded cross-validation searches.")
    parser.add_argument("--features", required=True, type=Path, help="Feature CSV containing label and numeric features")
    parser.add_argument("--model", required=True, type=Path, help="Joblib artifact to create")
    parser.add_argument("--report", required=True, type=Path, help="JSON tuning report to create")
    parser.add_argument("--test-size", type=float, default=0.2, help="Stratified test fraction (default: 0.2)")
    parser.add_argument("--random-state", type=int, default=42, help="Split, CV, and model seed (default: 42)")
    parser.add_argument("--cv-folds", type=int, default=3, help="Training folds for cross-validation (default: 3)")
    parser.add_argument("--max-configs", type=int, default=20, help="Maximum configurations per model (default: 20)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.features.exists() or not args.features.is_file():
        raise FileNotFoundError(f"Features file does not exist: {args.features}")
    artifact, report = tune_models(
        pd.read_csv(args.features),
        test_size=args.test_size,
        random_state=args.random_state,
        cv_folds=args.cv_folds,
        max_configs=args.max_configs,
    )
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
