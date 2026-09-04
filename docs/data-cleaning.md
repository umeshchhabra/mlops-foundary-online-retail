# Online Retail transaction cleaning

This is the second development step after dataset validation. It creates a
two auditable transaction views for later repeat-purchase label construction.
It does not define the 30-day target, create customer features, or train a
model.

The script reads the validated raw input and writes two new files: a cleaned
CSV files and a JSON audit summary. It never edits the raw workbook. Each
output row keeps its one-based source row number so an analyst can trace it
back to the input file.

The policy is intentionally narrow:

1. Require the eight Online Retail source columns.
2. Trim identifier and text fields and parse numeric/date fields.
3. Remove exact duplicate source rows, keeping the first copy.
4. Remove rows without `CustomerID`; the later target is customer-based.
5. Remove non-positive `UnitPrice` and unparseable required fields.
6. Retain cancellations, returns, and zero-quantity adjustments in the event
   view, classifying them as `cancellation`, `return`, or `adjustment`.
7. Build the purchase view from positive-quantity, non-cancellation events.
8. Retain rows with missing `Description`, because `StockCode` remains usable.

The event view is the source for later event-derived features such as return
or cancellation rates. The purchase view is the source for completed-purchase
labels and aggregates. The script records counts for every exclusion reason,
event type, input/output SHA-256 digests, and the exact rules used. It does not
call Airflow, MinIO, MLflow, Feast, or Kubernetes.

```powershell
python projects/retail-repeat-purchase/src/clean_transactions.py --help
python projects/retail-repeat-purchase/src/clean_transactions.py `
  --input "projects/retail-repeat-purchase/data/raw/Online Retail.xlsx" `
  --events-output projects/retail-repeat-purchase/data/processed/events.csv `
  --purchases-output projects/retail-repeat-purchase/data/processed/purchases.csv `
  --summary projects/retail-repeat-purchase/data/processed/cleaning-summary.json
```

The current source inspection found 541,909 input rows. The event and purchase
counts and their exclusion counts must be reviewed before we decide how to
construct historical customer snapshots and the future 30-day label.

## Recorded local run

The reviewed UCI workbook had SHA-256
`43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d`.
Running the command above produced 401,564 event rows and 392,692 purchase
rows. Events included 8,872 cancellations; purchases included 4,338
customers and 18,532 invoices. The base exclusions were 5,268 exact
duplicates, 135,037 rows without a customer ID, and 40 non-positive prices.
The event CSV had SHA-256
`b47a1b9260b3e9e1fa6b9bbc0368eb773ab993160cfbd302baf7c98ff24c3494`, and the
purchase CSV had SHA-256
`f2590d52459f71a90a74070b4e18437a2ff5605665f938830f71d0fdbdf14b28`. The
generated JSON records both output digests.
The generated JSON summary is intentionally ignored with the processed data;
rerunning the command recreates it and records these digests and counts.
