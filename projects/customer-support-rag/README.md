# Customer-support RAG

This project is the developer-side implementation of a production-oriented
retrieval-augmented generation pipeline. Infrastructure remains in the
separate platform repository.

## First dataset: MTRAG Government corpus

The first step uses the passage-level Government corpus from IBM's MTRAG
benchmark. The archive URL and expected SHA-256 are pinned in the download command below.
The archive and generated reports are local build artifacts and are intentionally not committed to Git.

Download and validate it locally:

```powershell
python projects/customer-support-rag/src/download_mtrag.py `
  --output projects/customer-support-rag/data/raw/govt.jsonl.zip `
  --sha256 09adcc5a1a8d11e362c66d7dbf0ef2ea338df4327d655102bf91ea8e258399f6

python projects/customer-support-rag/src/validate_mtrag.py `
  --input projects/customer-support-rag/data/raw/govt.jsonl.zip `
  --report projects/customer-support-rag/data/validation/mtrag-government.json `
  --expected-sha256 09adcc5a1a8d11e362c66d7dbf0ef2ea338df4327d655102bf91ea8e258399f6 `
  --expected-rows 49607
```

The validator checks the ZIP integrity, JSONL structure, required fields,
identifier uniqueness, URL shape, non-empty passage text, row count, and
content hashes. It does not embed or index data yet.

Create deterministic chunks after validation:

```powershell
python projects/customer-support-rag/src/chunk_mtrag.py `
  --input projects/customer-support-rag/data/raw/govt.jsonl.zip `
  --output projects/customer-support-rag/data/processed/mtrag-government-chunks.jsonl
```

Each chunk keeps its document ID, source URL, title, chunk position, and a
stable chunk ID. The default 1,200-character window and 200-character overlap
are recorded in `params.yaml`.

## Reproduce with DVC

`dvc.yaml` defines three reproducible stages: download the pinned archive,
validate it, and create deterministic chunks. Run the full pipeline from the
repository root:

```powershell
.\.venv\Scripts\dvc.exe repro projects/customer-support-rag/dvc.yaml
```

The run creates `dvc.lock` with the source, parameter, dependency, and output
hashes. Commit that lock file, but keep the data and generated reports ignored
by Git. After the dedicated MinIO bucket is provisioned and DVC credentials are
configured, upload the cached outputs:

```powershell
.\.venv\Scripts\dvc.exe push projects/customer-support-rag/dvc.yaml
```

A fresh checkout can restore the exact locked outputs with:

```powershell
.\.venv\Scripts\dvc.exe pull projects/customer-support-rag/dvc.yaml
```

The configured remote is the project bucket `customer-support-rag-data`; no
bucket or infrastructure resources are created by this repository.

Run the focused tests with:

```powershell
python -m unittest discover projects/customer-support-rag/tests -p 'test_*.py'
```

Customer records will be synthetic during development. Approved private data
will later be accessed through an authorization-aware service rather than
placed in the public document index.
