# Local baseline training

This is the first modeling step. It consumes the feature table and compares
three deterministic classifiers without calling Airflow, MLflow, Feast,
KServe, or Kubernetes.

The script excludes snapshot identifiers and timestamps from model inputs. It
uses a stratified 80/20 split with a fixed seed, fits a dummy classifier,
standardized logistic regression, and random forest, then selects the model
with the highest test macro-F1 (accuracy breaks ties). The selected estimator
and its feature-column order are saved together in a Joblib artifact.

```powershell
python -m pip install -r requirements/training.txt
python projects/retail-repeat-purchase/src/train_baselines.py --help
python projects/retail-repeat-purchase/src/train_baselines.py `
  --features projects/retail-repeat-purchase/data/processed/features.csv `
  --model projects/retail-repeat-purchase/models/baseline/model.joblib `
  --report projects/retail-repeat-purchase/reports/baseline-report.json
```

The report contains the input and model hashes, split sizes, feature columns,
per-model accuracy, macro-F1, per-class metrics, and confusion matrices. The
model artifact is ignored by Git; the report can be reviewed before later
experiment tracking or promotion work.

This baseline uses one existing cutoff snapshot and a customer-level random
split. A later modeling step can add multiple temporal cutoffs and a temporal
holdout once that evaluation policy is agreed.

Run tests with:

```powershell
python -m unittest discover projects/retail-repeat-purchase/tests
```
