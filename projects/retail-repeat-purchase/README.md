# Retail repeat-purchase project

This directory contains the customer repeat-purchase project. Dataset
validation is implemented in `pipelines/shared/`; this branch adds the next
step, a documented transaction-cleaning script. Label construction, feature
store, and model training are still separate later steps.

## Clean transactions

The current policy keeps positive purchase lines for identified customers,
removes cancellation invoices, returns/adjustments, non-positive prices, and
exact duplicate rows, and retains missing descriptions. Every removed row is
counted by reason. The raw file is never modified.

```powershell
python projects/retail-repeat-purchase/src/clean_transactions.py `
  --input "projects/retail-repeat-purchase/data/raw/Online Retail.xlsx" `
  --output projects/retail-repeat-purchase/data/processed/transactions.csv `
  --summary projects/retail-repeat-purchase/data/processed/cleaning-summary.json
```

Run the unit tests with `python -m unittest discover projects/retail-repeat-purchase/tests`.

The intended target is a documented prediction window such as: whether a
customer makes another purchase within 30 days after a historical cutoff. That
definition and the treatment of cancellations, returns, missing customer IDs,
and duplicate transactions must be agreed before implementation.
