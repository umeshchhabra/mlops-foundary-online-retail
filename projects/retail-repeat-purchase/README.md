# Retail repeat-purchase project

This directory contains the customer repeat-purchase project. Dataset
validation, transaction cleaning, and label construction are implemented in
the preceding steps. Feature store and model training are still separate later
steps.

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
