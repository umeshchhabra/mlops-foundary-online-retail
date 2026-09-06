# Model evaluation and packaging

This step turns the tuned local artifact into a checked package. It requires
the selected model's held-out macro-F1 to improve on the dummy baseline, checks
that the feature CSV hash matches the tuning report, and verifies that every
declared feature is numeric and non-null.

The packaged Joblib file includes the fitted estimator, feature order, input
schema, dataset/report/training-code hashes, dependency metadata from tuning,
and the evaluation scores. A manifest records the package hash and a
deterministic prediction hash for the complete input table.

Run it after model tuning:

```powershell
python projects/retail-repeat-purchase/src/package_model.py `
  --model projects/retail-repeat-purchase/models/tuned/model.joblib `
  --report projects/retail-repeat-purchase/reports/tuning-report.json `
  --features projects/retail-repeat-purchase/data/processed/features.csv `
  --code projects/retail-repeat-purchase/src/tune_baselines.py `
  --output projects/retail-repeat-purchase/models/tuned/model-package.joblib `
  --manifest projects/retail-repeat-purchase/reports/model-package.json
```

The command is local and does not modify infrastructure or deploy a model.
The package manifest is the review point for the next experiment-tracking or
serving step.
