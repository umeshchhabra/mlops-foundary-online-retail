"""Download a tracked model package and verify local prediction parity."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from package_model import sha256


ARTIFACTS = {
    "model": "model/model-package.joblib",
    "manifest": "reports/model-package.json",
    "report": "reports/tuning-report.json",
}


def _prediction_hash(predictions: np.ndarray) -> str:
    encoded = json.dumps(predictions.tolist(), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_package(
    local_model_path: Path,
    downloaded_model_path: Path,
    manifest_path: Path,
    report_path: Path,
    features_path: Path,
) -> dict:
    """Verify artifact hashes, input schema, and prediction parity."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_model_hash = manifest["model"]["sha256"]
    if sha256(local_model_path) != expected_model_hash:
        raise ValueError("Local model hash does not match the package manifest")
    if sha256(downloaded_model_path) != expected_model_hash:
        raise ValueError("Downloaded model hash does not match the package manifest")
    if sha256(features_path) != manifest["dataset_sha256"]:
        raise ValueError("Feature input hash does not match the package manifest")
    if sha256(report_path) != manifest["report_sha256"]:
        raise ValueError("Downloaded tuning report hash does not match the package manifest")

    local_package = joblib.load(local_model_path)
    downloaded_package = joblib.load(downloaded_model_path)
    columns = downloaded_package.get("feature_columns", [])
    if not columns or columns != local_package.get("feature_columns"):
        raise ValueError("Local and downloaded model feature orders do not match")
    if downloaded_package.get("input_schema") != manifest.get("input_schema"):
        raise ValueError("Downloaded model schema does not match the package manifest")

    frame = pd.read_csv(features_path)
    for column in columns:
        if column not in frame or not is_numeric_dtype(frame[column]) or frame[column].isna().any():
            raise ValueError(f"Feature input violates the model schema: {column}")
        expected_dtype = manifest["input_schema"][column]["dtype"]
        if str(frame[column].dtype) != expected_dtype:
            raise ValueError(f"Feature input dtype does not match for {column}")

    local_predictions = np.asarray(local_package["model"].predict(frame[columns]))
    downloaded_predictions = np.asarray(downloaded_package["model"].predict(frame[columns]))
    if not np.array_equal(local_predictions, downloaded_predictions):
        raise ValueError("Downloaded model predictions do not match the local package")
    prediction_hash = _prediction_hash(downloaded_predictions)
    if prediction_hash != manifest["prediction_check"]["prediction_sha256"]:
        raise ValueError("Prediction hash does not match the package manifest")
    return {
        "model_sha256": expected_model_hash,
        "dataset_sha256": manifest["dataset_sha256"],
        "prediction_sha256": prediction_hash,
        "rows": int(len(downloaded_predictions)),
        "prediction_parity": True,
        "schema_verified": True,
    }


def download_run(tracking_uri: str, run_id: str, destination: Path):
    try:
        from mlflow.tracking import MlflowClient
    except ImportError as exc:
        raise RuntimeError("Install requirements/tracking.txt before downloading artifacts") from exc
    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    if run.info.status != "FINISHED":
        raise ValueError(f"MLflow run must be FINISHED, found {run.info.status}")
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        name: Path(client.download_artifacts(run_id, artifact, str(destination)))
        for name, artifact in ARTIFACTS.items()
    }
    return run, paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download an MLflow model package and verify prediction parity.")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    parser.add_argument("--run-id", required=True, help="Finished MLflow run ID")
    parser.add_argument("--download-dir", required=True, type=Path, help="Local destination for downloaded artifacts")
    parser.add_argument("--local-model", required=True, type=Path, help="Original local model package")
    parser.add_argument("--features", required=True, type=Path, help="Feature CSV used for parity checking")
    parser.add_argument("--result", required=True, type=Path, help="Verification result JSON")
    parser.add_argument("--s3-endpoint-url", default=os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"))
    parser.add_argument("--access-key-id", default=os.getenv("AWS_ACCESS_KEY_ID"), help="S3 access key (prefer environment variable)")
    parser.add_argument("--secret-access-key", default=os.getenv("AWS_SECRET_ACCESS_KEY"), help="S3 secret key (prefer environment variable)")
    parser.add_argument("--region", default=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    for path in (args.local_model, args.features):
        if not path.is_file():
            raise FileNotFoundError(f"Required file does not exist: {path}")
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = args.s3_endpoint_url
    if args.access_key_id:
        os.environ["AWS_ACCESS_KEY_ID"] = args.access_key_id
    if args.secret_access_key:
        os.environ["AWS_SECRET_ACCESS_KEY"] = args.secret_access_key
    os.environ["AWS_DEFAULT_REGION"] = args.region
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    run, paths = download_run(args.tracking_uri, args.run_id, args.download_dir)
    result = verify_package(args.local_model, paths["model"], paths["manifest"], paths["report"], args.features)
    tags = run.data.tags
    if tags.get("model_package_sha256") != result["model_sha256"]:
        raise ValueError("MLflow model hash tag does not match the downloaded package")
    if tags.get("dataset_sha256") != result["dataset_sha256"]:
        raise ValueError("MLflow dataset hash tag does not match the package manifest")
    result.update({
        "tracking_uri": args.tracking_uri,
        "run_id": args.run_id,
        "artifact_uri": run.info.artifact_uri,
        "downloaded_artifacts": list(ARTIFACTS.values()),
    })
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
