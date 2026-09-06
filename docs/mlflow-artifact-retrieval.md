# MLflow artifact retrieval and parity

This step downloads the model package, package manifest, and tuning report
from a finished MLflow run. It verifies the recorded model, dataset, and report
hashes, checks the numeric feature schema, and requires the downloaded model to
produce exactly the same predictions as the original local package.

Install `requirements/tracking.txt`, provide the existing MinIO credentials,
and run from the repository root:

```powershell
$env:MLFLOW_S3_ENDPOINT_URL = "http://localhost:9000"
$env:AWS_ACCESS_KEY_ID = "<MinIO user>"
$env:AWS_SECRET_ACCESS_KEY = "<MinIO password>"

python projects/retail-repeat-purchase/src/retrieve_mlflow.py `
  --run-id <run-id> `
  --download-dir projects/retail-repeat-purchase/models/downloaded/<run-id> `
  --local-model projects/retail-repeat-purchase/models/tuned/model-package.joblib `
  --features projects/retail-repeat-purchase/data/processed/features.csv `
  --result projects/retail-repeat-purchase/reports/mlflow-retrieval.json
```

Use `http://minio.mlops.svc.cluster.local:9000` instead when the client runs
inside Kubernetes. Downloaded model files stay under the ignored `models/`
directory; the JSON verification result can be reviewed and committed.
