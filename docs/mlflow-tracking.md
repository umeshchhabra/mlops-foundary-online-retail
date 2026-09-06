# MLflow experiment tracking

This step logs the packaged local model to the existing MLflow tracking
service. It records tuning parameters, cross-validation and held-out metrics,
dataset/model/report hashes, and uploads the Joblib package, package manifest,
and tuning report as artifacts.

The `retail-repeat-purchase` experiment stores artifacts in
`s3://retail-repeat-purchase-artifacts`.

Install the developer tracking dependency:

```powershell
python -m pip install -r requirements/tracking.txt
```

Then run from the repository root. The default tracking URI is
`http://localhost:5000`; set `MLFLOW_TRACKING_URI` or pass `--tracking-uri`
when the existing service uses another address.

```powershell
python projects/retail-repeat-purchase/src/track_mlflow.py `
  --model projects/retail-repeat-purchase/models/tuned/model-package.joblib `
  --report projects/retail-repeat-purchase/reports/tuning-report.json `
  --manifest projects/retail-repeat-purchase/reports/model-package.json `
  --result projects/retail-repeat-purchase/reports/mlflow-run.json
```

For the MinIO-backed artifact store, provide the existing endpoint and
credentials through environment variables (or the equivalent CLI options):

```powershell
$env:MLFLOW_S3_ENDPOINT_URL = "http://localhost:9000"
$env:AWS_ACCESS_KEY_ID = "<MinIO user>"
$env:AWS_SECRET_ACCESS_KEY = "<MinIO password>"
$env:AWS_DEFAULT_REGION = "us-east-1"
```

The command creates one run in the `retail-repeat-purchase` experiment and
prints its run ID. It does not deploy or promote the model and does not modify
infrastructure.
