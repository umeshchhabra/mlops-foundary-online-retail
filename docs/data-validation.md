# Dataset validation

Run validation before cleaning, feature engineering, or training. The script
is deliberately independent of Airflow, Feast, MLflow, and Kubernetes so that
data problems are found before platform work begins.

```powershell
python pipelines/shared/dataset_validation.py --help
```

Supported inputs are CSV, XLSX/XLS, Parquet, JSON, and JSONL. The command is
read-only and returns exit code `0` when the file is readable, non-empty, and
meets the requested checks. It returns `1` for a missing/unreadable file or a
failed expectation. Missing values and duplicates are warnings by default;
`--fail-on-warning` makes them fail the command.

The report includes file size and SHA-256, parse status, row and column counts,
ordered column names, inferred dtypes, duplicate column names, missing-cell
counts, duplicate rows, non-finite numeric values, date parsing failures and
ranges, required/expected schema failures, and JSON-safe sample rows.

Use `--expected-sha256` to verify a known dataset version. The digest is always
reported, even when no expected digest is supplied. Repeat `--required-column`
and `--date-column` for multiple columns. Use `--max-size-mb` as a guard against
an unexpectedly large input.

## Online Retail baseline inspection

The UCI Online Retail workbook is expected to contain 541,909 rows and these
eight columns:

`InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country`

The baseline inspection found a readable file with no invalid dates or
non-finite numeric values. It also found missing `CustomerID` and `Description`
values and duplicate rows. Invoice cancellations and negative quantities are
business data characteristics, not file corruption. Their treatment belongs
to the next, separately reviewed cleaning step.
