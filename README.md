# MLOps Foundry — Customer-support RAG

This repository contains the developer-side implementation of a production
RAG pipeline. Kubernetes, Helm, Argo CD, Airflow, MinIO, MLflow, Redis, KServe,
and observability services are maintained in the separate infrastructure
repository.

## Current project

The active developer project is
`projects/customer-support-rag/`. It starts with the pinned IBM MTRAG
Government passage corpus and implements validation and deterministic chunking.

```powershell
python projects/customer-support-rag/src/download_mtrag.py `
  --output projects/customer-support-rag/data/raw/govt.jsonl.zip `
  --sha256 09adcc5a1a8d11e362c66d7dbf0ef2ea338df4327d655102bf91ea8e258399f6

python projects/customer-support-rag/src/validate_mtrag.py `
  --input projects/customer-support-rag/data/raw/govt.jsonl.zip `
  --report projects/customer-support-rag/data/validation/mtrag-government.json `
  --expected-sha256 09adcc5a1a8d11e362c66d7dbf0ef2ea338df4327d655102bf91ea8e258399f6 `
  --expected-rows 49607

python projects/customer-support-rag/src/chunk_mtrag.py `
  --input projects/customer-support-rag/data/raw/govt.jsonl.zip `
  --output projects/customer-support-rag/data/processed/mtrag-government-chunks.jsonl
```

Run the RAG tests with:

```powershell
python -m unittest discover projects/customer-support-rag/tests -p 'test_*.py'
```

See [the dataset foundation guide](docs/rag-dataset-foundation.md) for the
immutable source manifest and the required dedicated MinIO bucket.
