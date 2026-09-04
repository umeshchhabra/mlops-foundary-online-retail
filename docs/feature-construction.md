# Leakage-safe feature construction

This is the next development step after label construction. It creates one
feature row for every customer/cutoff snapshot in `labels.csv`. It does not
train a model or call Airflow, Feast, MLflow, or Kubernetes.

## Inputs and leakage boundary

The builder reads both cleaned views:

- `purchases.csv` supplies completed-purchase aggregates.
- `events.csv` supplies event counts and cancellation/return signals.

For every snapshot, all aggregates use only rows satisfying:

```text
invoice_date >= observation_start and invoice_date < cutoff_date
```

The prediction window is never read. The output retains the label for the
later training step, but does not include future purchase counts.

## Features

Purchase features include purchase-line count, distinct invoices, distinct
stock codes, total quantity and revenue, active purchase days, recency,
average order value, and average items per invoice. Event features include
event count, cancellation/return/adjustment counts, cancellation and return
rates, and net event revenue.

## Run it

```powershell
python projects/retail-repeat-purchase/src/build_features.py --help
python projects/retail-repeat-purchase/src/build_features.py `
  --events projects/retail-repeat-purchase/data/processed/events.csv `
  --purchases projects/retail-repeat-purchase/data/processed/purchases.csv `
  --labels projects/retail-repeat-purchase/data/processed/labels.csv `
  --output projects/retail-repeat-purchase/data/processed/features.csv `
  --summary projects/retail-repeat-purchase/data/processed/features-summary.json
```

The JSON summary records input/output hashes and null counts. The processed
outputs are ignored by Git and can be recreated from the documented inputs.

Run tests with:

```powershell
python -m unittest discover projects/retail-repeat-purchase/tests
```

## Recorded local run

Using the merged event, purchase, and label inputs produced 2,566 feature rows
for the cutoff `2011-11-09T00:00:00`. The output had 21 columns, no null
feature values, and the same 1,103/1,463 label class counts as the label
input. The generated feature CSV had SHA-256
`742e55f68f9c9af8047ea4b4ee332b01d4e5e119cc00fcbf111973861e8fe9ee`.
