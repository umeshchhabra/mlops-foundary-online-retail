# RAG dataset foundation

The developer project starts with IBM's MTRAG Government passage corpus. It is
an evaluation and retrieval corpus, not private customer data. The immutable
source manifest records the URL, archive hash, member name, schema, and observed
row counts.

## Infrastructure needed before versioned storage

The developer repository needs a dedicated MinIO bucket for this project, such
as `customer-support-rag-data`. It should not reuse the retail DVC bucket. The
infrastructure repository should provide a scoped credential for the bucket.

The next DVC step will store the archive and later processed chunk data under
prefixes such as:

```text
raw/mtrag-government/
processed/chunks/
manifests/
```

No vector database, embedding endpoint, model endpoint, or Kubernetes change is
needed for this validation-only step.
