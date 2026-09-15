# RAG dataset foundation

The developer project starts with IBM's MTRAG Government passage corpus. It is
an evaluation and retrieval corpus, not private customer data. The source URL
is pinned in `src/download_mtrag.py`, and `params.yaml` records the upstream
commit, archive checksum, member name, schema expectations, and row count.

## DVC pipeline

`projects/customer-support-rag/dvc.yaml` contains three stages:

1. `download` fetches the immutable upstream archive and verifies its SHA-256.
2. `validate` checks ZIP integrity, schema, IDs, URLs, text, and expected rows.
3. `chunk` creates deterministic overlapping passage chunks.

The checked-in `projects/customer-support-rag/dvc.lock` records the exact
parameters, source-code hashes, and output hashes observed when the pipeline
was run. The archive, validation report, and chunk file remain ignored local
artifacts.

Run the pipeline from the repository root:

```powershell
.\.venv\Scripts\dvc.exe repro projects/customer-support-rag/dvc.yaml
```

A clean checkout with data already uploaded to the DVC remote can restore the
locked outputs without downloading from the upstream source:

```powershell
.\.venv\Scripts\dvc.exe pull projects/customer-support-rag/dvc.yaml
```

Use `dvc push` after a successful reproduction to publish the cache objects:

```powershell
.\.venv\Scripts\dvc.exe push projects/customer-support-rag/dvc.yaml
```

## Infrastructure contract

The developer repository expects a dedicated MinIO bucket named
`customer-support-rag-data`, configured in `.dvc/config`. The infrastructure
repository should provide a scoped access key and secret for that bucket. A
path-style S3 endpoint at `http://localhost:9000` is used for local host
execution; Airflow or other in-cluster clients should use the cluster-internal
MinIO service endpoint supplied by the infrastructure repository.

Suggested object prefixes are:

```text
raw/mtrag-government/
processed/chunks/
manifests/
```

The developer workflow does not create buckets, credentials, vector stores,
embedding endpoints, model endpoints, or Kubernetes resources.

## Verification

The focused unit tests cover archive validation and deterministic chunking:

```powershell
python -m unittest discover projects/customer-support-rag/tests -p 'test_*.py'
```

To inspect the planned commands without executing them, use:

```powershell
.\.venv\Scripts\dvc.exe repro --dry projects/customer-support-rag/dvc.yaml
```
