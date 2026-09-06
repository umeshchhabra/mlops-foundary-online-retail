# Bounded model tuning

This step tunes the two useful baseline classifiers while keeping the dummy
classifier as a test-set reference. A stratified 20% test split is created
once. Only the remaining 80% is searched with shuffled three-fold
cross-validation, so the test rows are not used to choose parameters.

The logistic regression search has six configurations (`C` and class weight).
The random forest search has twelve configurations (tree count, depth, and
minimum leaf size). Both searches use one worker and a fixed seed. The model
with the highest cross-validation macro-F1 is selected; cross-validation
accuracy breaks an exact tie.

Run the command from the repository root:

```powershell
python projects/retail-repeat-purchase/src/tune_baselines.py `
  --features projects/retail-repeat-purchase/data/processed/features.csv `
  --model projects/retail-repeat-purchase/models/tuned/model.joblib `
  --report projects/retail-repeat-purchase/reports/tuning-report.json
```

The report includes the dummy test metrics, each model's selected parameters,
configuration count, cross-validation mean and spread, and one held-out test
evaluation with per-class metrics and a confusion matrix. The Joblib artifact
contains the fitted selected estimator, feature order, selected parameters,
split settings, and dependency versions.

This remains a local, reproducible modeling step. It does not call Airflow,
MLflow, Feast, KServe, or Kubernetes. Promotion and serving are separate
steps after the evaluation policy is agreed.
