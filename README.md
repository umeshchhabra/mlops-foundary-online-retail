# MLOps Foundry — Online Retail

This repository contains the development workflow for the Online Retail
repeat-purchase project. Infrastructure lives in the separate MLOps Foundry
repository; this repository contains data and model code, shared pipeline
helpers, feature definitions, deployment descriptors, and documentation.

## First step: validate a dataset

Validation is read-only. It checks that a file can be read, records its SHA-256
digest, inspects dimensions and schema, reports missing values and duplicates,
checks numeric values and requested date columns, and emits a JSON report. It
does not clean or rewrite the input.

```powershell
python -m pip install -r requirements/validation.txt
python pipelines/shared/dataset_validation.py --help
python pipelines/shared/dataset_validation.py `
  --input "projects/retail-repeat-purchase/data/raw/Online Retail.xlsx" `
  --required-column CustomerID `
  --required-column InvoiceDate `
  --date-column InvoiceDate `
  --expected-columns "InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country" `
  --report projects/retail-repeat-purchase/data/validation/online-retail.json
```

The raw dataset and generated reports are ignored by Git. Commit the expected
digest and validation metadata only after reviewing them.

See [the validation guide](docs/data-validation.md) for the checks, exit codes,
and the interpretation of the current Online Retail source.
