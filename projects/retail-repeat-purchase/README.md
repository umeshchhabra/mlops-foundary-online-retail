# Retail repeat-purchase project

This directory contains the customer repeat-purchase project. Dataset
validation, transaction cleaning, and label construction are implemented in
the preceding steps. Feature store remains a separate later step.

## Train local baselines

The baseline trainer compares a dummy classifier, standardized logistic
regression, and random forest with a deterministic stratified split. It saves
the selected estimator and feature order together in a local Joblib artifact.

```powershell
python -m pip install -r requirements/training.txt
python projects/retail-repeat-purchase/src/train_baselines.py `
  --features projects/retail-repeat-purchase/data/processed/features.csv `
  --model projects/retail-repeat-purchase/models/baseline/model.joblib `
  --report projects/retail-repeat-purchase/reports/baseline-report.json
```

See `docs/model-baseline.md` for the evaluation contract and report details.

## Build leakage-safe features

Feature construction uses only each label snapshot's observation window. It
combines completed-purchase aggregates with cancellation and return signals
from the event view.

```powershell
python projects/retail-repeat-purchase/src/build_features.py `
  --events projects/retail-repeat-purchase/data/processed/events.csv `
  --purchases projects/retail-repeat-purchase/data/processed/purchases.csv `
  --labels projects/retail-repeat-purchase/data/processed/labels.csv `
  --output projects/retail-repeat-purchase/data/processed/features.csv `
  --summary projects/retail-repeat-purchase/data/processed/features-summary.json
```

The feature definition and leakage boundary are documented in
`docs/feature-construction.md`.

## Build repeat-purchase labels

Labels use 90 days of completed-purchase history and a 30-day prediction
window. A customer is eligible only when they purchased during the observation
window; label `1` means they purchase at least once in the following window.
Both windows are half-open, so the cutoff belongs to the prediction window.

```powershell
python projects/retail-repeat-purchase/src/build_labels.py `
  --purchases projects/retail-repeat-purchase/data/processed/purchases.csv `
  --output projects/retail-repeat-purchase/data/processed/labels.csv `
  --summary projects/retail-repeat-purchase/data/processed/labels-summary.json `
  --cutoff 2011-11-09T00:00:00
```

Repeat `--cutoff` to create multiple temporal snapshots. The command rejects
partial history or future windows, invalid purchases, and cancellation events
so labels cannot silently use censored or leaked data.

## Clean transactions

The script creates two views for identified customers: `events.csv` keeps
valid purchases, cancellations, returns, and zero-quantity adjustments;
`purchases.csv` keeps only positive non-cancellation purchase lines. Exact
duplicates, invalid required fields, and non-positive prices are excluded and
counted. The raw file is never modified.

```powershell
python projects/retail-repeat-purchase/src/clean_transactions.py `
  --input "projects/retail-repeat-purchase/data/raw/Online Retail.xlsx" `
  --events-output projects/retail-repeat-purchase/data/processed/events.csv `
  --purchases-output projects/retail-repeat-purchase/data/processed/purchases.csv `
  --summary projects/retail-repeat-purchase/data/processed/cleaning-summary.json
```

Run the unit tests with `python -m unittest discover projects/retail-repeat-purchase/tests`.

The label definition and leakage rules are documented in
`docs/label-construction.md`. Feature construction is the next separate step.

## Tune model parameters

Tune logistic regression and random forest with bounded three-fold cross-validation while keeping the dummy classifier as a held-out test reference.

```powershell
python projects/retail-repeat-purchase/src/tune_baselines.py `
  --features projects/retail-repeat-purchase/data/processed/features.csv `
  --model projects/retail-repeat-purchase/models/tuned/model.joblib `
  --report projects/retail-repeat-purchase/reports/tuning-report.json
```

See `docs/model-tuning.md` for the search and evaluation contract.

## Package the tuned model

Validate the tuned report, attach the feature schema and lineage hashes, and run a deterministic local prediction check.

```powershell
python projects/retail-repeat-purchase/src/package_model.py `
  --model projects/retail-repeat-purchase/models/tuned/model.joblib `
  --report projects/retail-repeat-purchase/reports/tuning-report.json `
  --features projects/retail-repeat-purchase/data/processed/features.csv `
  --code projects/retail-repeat-purchase/src/tune_baselines.py `
  --output projects/retail-repeat-purchase/models/tuned/model-package.joblib `
  --manifest projects/retail-repeat-purchase/reports/model-package.json
```

See `docs/model-packaging.md` for the package contract.

## Track the packaged model in MLflow

Install `requirements/tracking.txt`, then log the package, manifest, tuning report, parameters, metrics, and lineage hashes to the existing MLflow service.

```powershell
python projects/retail-repeat-purchase/src/track_mlflow.py `
  --model projects/retail-repeat-purchase/models/tuned/model-package.joblib `
  --report projects/retail-repeat-purchase/reports/tuning-report.json `
  --manifest projects/retail-repeat-purchase/reports/model-package.json `
  --result projects/retail-repeat-purchase/reports/mlflow-run.json
```

See `docs/mlflow-tracking.md` for the tracking contract.

## Retrieve and verify the MLflow artifact

Download the tracked package and reports, verify their hashes and schema, and confirm prediction parity with the local model package.

```powershell
python projects/retail-repeat-purchase/src/retrieve_mlflow.py `
  --run-id <run-id> `
  --download-dir projects/retail-repeat-purchase/models/downloaded/<run-id> `
  --local-model projects/retail-repeat-purchase/models/tuned/model-package.joblib `
  --features projects/retail-repeat-purchase/data/processed/features.csv `
  --result projects/retail-repeat-purchase/reports/mlflow-retrieval.json
```

See `docs/mlflow-artifact-retrieval.md` for connection settings and verification details.

## Version the training inputs with DVC

The raw workbook and exact processed feature table are tracked by DVC and stored in the MinIO bucket `retail-repeat-purchase-data`.

```powershell
python -m pip install -r requirements/dvc.txt
python -m dvc pull
python -m dvc status
```

See `docs/dvc-data-versioning.md` for credentials and restore verification.
