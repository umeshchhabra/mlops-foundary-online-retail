"""Log the packaged retail model and evaluation results to MLflow."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _value(value) -> str | int | float:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return value
    return str(value)


def build_run_payload(report: dict, manifest: dict, model_path: Path, report_path: Path, manifest_path: Path) -> dict:
    """Build MLflow-safe params, metrics, tags, and artifacts without network calls."""
    selected = report["selected_model"]
    tuned = report["models"][selected]
    params = {
        "selected_model": selected,
        "random_state": report["random_state"],
        "test_size": report["test_size"],
        "cv_folds": report["cv_folds"],
        "scoring": report["scoring"],
        "feature_count": len(report["feature_columns"]),
    }
    params.update({f"selected_{key}": value for key, value in tuned["best_params"].items()})
    metrics = {
        "baseline_test_macro_f1": report["baseline"]["test"]["macro_f1"],
        "selected_cv_macro_f1": tuned["cv_macro_f1"],
        "selected_cv_macro_f1_std": tuned["cv_macro_f1_std"],
        "selected_test_macro_f1": tuned["test"]["macro_f1"],
        "selected_test_accuracy": tuned["test"]["accuracy"],
    }
    tags = {
        "project": "retail-repeat-purchase",
        "workflow": "local-model-packaging",
        "dataset_sha256": manifest["dataset_sha256"],
        "model_package_sha256": manifest["model"]["sha256"],
        "report_sha256": manifest["report_sha256"],
    }
    return {
        "params": {key: _value(value) for key, value in params.items()},
        "metrics": {key: float(value) for key, value in metrics.items()},
        "tags": tags,
        "artifacts": [(model_path, "model"), (manifest_path, "reports"), (report_path, "reports")],
    }


def track_run(
    tracking_uri: str,
    experiment: str,
    run_name: str,
    payload: dict,
) -> str:
    """Create one MLflow run using the installed MLflow SDK."""
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("Install requirements/tracking.txt before tracking a run") from exc
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name, tags=payload["tags"]) as run:
        mlflow.log_params(payload["params"])
        mlflow.log_metrics(payload["metrics"])
        for path, artifact_path in payload["artifacts"]:
            mlflow.log_artifact(str(path), artifact_path=artifact_path)
        return run.info.run_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Log a packaged retail model to MLflow.")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    parser.add_argument("--experiment", default="retail-repeat-purchase")
    parser.add_argument("--run-name", default="tuned-model-package")
    parser.add_argument("--model", required=True, type=Path, help="Packaged Joblib artifact")
    parser.add_argument("--report", required=True, type=Path, help="Tuning report JSON")
    parser.add_argument("--manifest", required=True, type=Path, help="Model package manifest JSON")
    parser.add_argument("--result", type=Path, help="Optional JSON file for the created run ID")
    parser.add_argument("--s3-endpoint-url", default=os.getenv("MLFLOW_S3_ENDPOINT_URL"), help="MinIO/S3 endpoint for artifact uploads")
    parser.add_argument("--access-key-id", default=os.getenv("AWS_ACCESS_KEY_ID"), help="S3 access key (prefer environment variable)")
    parser.add_argument("--secret-access-key", default=os.getenv("AWS_SECRET_ACCESS_KEY"), help="S3 secret key (prefer environment variable)")
    parser.add_argument("--region", default=os.getenv("AWS_DEFAULT_REGION", "us-east-1"), help="S3 region")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    for path in (args.model, args.report, args.manifest):
        if not path.is_file():
            raise FileNotFoundError(f"Required file does not exist: {path}")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    payload = build_run_payload(report, manifest, args.model, args.report, args.manifest)
    if args.s3_endpoint_url:
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = args.s3_endpoint_url
    if args.access_key_id:
        os.environ["AWS_ACCESS_KEY_ID"] = args.access_key_id
    if args.secret_access_key:
        os.environ["AWS_SECRET_ACCESS_KEY"] = args.secret_access_key
    os.environ["AWS_DEFAULT_REGION"] = args.region
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run_id = track_run(args.tracking_uri, args.experiment, args.run_name, payload)
    result = {"tracking_uri": args.tracking_uri, "experiment": args.experiment, "run_id": run_id}
    if args.result:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
