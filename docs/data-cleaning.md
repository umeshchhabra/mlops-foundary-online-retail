# Online Retail transaction cleaning

This is the second development step after dataset validation. It creates a
clean transaction table for later repeat-purchase label construction. It does
not define the 30-day target, create customer features, or train a model.

The script reads the validated raw input and writes two new files: a cleaned
CSV and a JSON audit summary. It never edits the raw workbook. Each output row
keeps its one-based source row number so an analyst can trace it back to the
input file.

The policy is intentionally narrow:

1. Require the eight Online Retail source columns.
2. Trim identifier and text fields and parse numeric/date fields.
3. Remove exact duplicate source rows, keeping the first copy.
4. Remove rows without `CustomerID`; the later target is customer-based.
5. Remove cancellation invoices whose `InvoiceNo` starts with `C`.
6. Remove non-positive `Quantity`, which represents returns or adjustments for
   this purchase-history view.
7. Remove non-positive `UnitPrice` and unparseable required identifiers/dates.
8. Retain rows with missing `Description`, because `StockCode` remains usable.

The script records counts for every exclusion reason, input/output SHA-256
digests, row and customer/invoice counts, date range, retained revenue, and
the exact rules used. It does not call Airflow, MinIO, MLflow, Feast, or
Kubernetes.

```powershell
python projects/retail-repeat-purchase/src/clean_transactions.py --help
python projects/retail-repeat-purchase/src/clean_transactions.py `
  --input "projects/retail-repeat-purchase/data/raw/Online Retail.xlsx" `
  --output projects/retail-repeat-purchase/data/processed/transactions.csv `
  --summary projects/retail-repeat-purchase/data/processed/cleaning-summary.json
```

The current source inspection found 541,909 input rows. The cleaning output
and its exclusion counts must be reviewed before we decide how to construct
historical customer snapshots and the future 30-day repeat-purchase label.

## Recorded local run

The reviewed UCI workbook had SHA-256
`43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d`.
Running the command above retained 392,692 rows, 4,338 customers, and 18,532
invoices. It excluded 8,872 cancellations, 5,268 exact duplicates, 135,037
rows without a customer ID, and 40 non-positive prices. The cleaned CSV had
SHA-256 `d23289b23319ff3e209108839552656f6db56d276e7033e528f5e35b96c57803`.
The generated JSON summary is intentionally ignored with the processed data;
rerunning the command recreates it and records these digests and counts.
