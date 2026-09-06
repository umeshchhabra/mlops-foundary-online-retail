# DVC data versioning

This project uses DVC for the two inputs that define a training dataset:

- `projects/retail-repeat-purchase/data/raw/Online Retail.xlsx`
- `projects/retail-repeat-purchase/data/processed/features.csv`

The files remain out of Git. Their small `.dvc` pointer files are committed,
and the content is stored in the MinIO bucket `retail-repeat-purchase-data`.
The default DVC remote records only the bucket location in `.dvc/config`;
credentials stay in environment variables or local machine configuration.

Install the developer dependency and configure the local endpoint:

```powershell
python -m pip install -r requirements/dvc.txt
$env:AWS_ACCESS_KEY_ID = "<MinIO user>"
$env:AWS_SECRET_ACCESS_KEY = "<MinIO password>"
$env:AWS_DEFAULT_REGION = "us-east-1"
```

The committed pointers can be restored with:

```powershell
python -m dvc pull
```

Use `python -m dvc status` to check whether the working files match their
pointers and `python -m dvc push` after changing a tracked input. The exact
feature-table hash is recorded in downstream model manifests, so an Airflow or
local training run can prove which DVC input it consumed.

The MinIO endpoint is `http://localhost:9000` from Windows, or
`http://minio.mlops.svc.cluster.local:9000` from Kubernetes. The DVC pointers
are intentionally separate from the later Airflow and Feast steps.
