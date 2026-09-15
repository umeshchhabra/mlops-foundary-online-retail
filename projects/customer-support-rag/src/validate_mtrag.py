import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse


REQUIRED_FIELDS = ("_id", "id", "url", "text", "title")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_archive(archive: Path, member: str = "govt.jsonl") -> dict:
    if not archive.is_file():
        raise ValueError(f"Archive does not exist: {archive}")

    missing_fields = Counter()
    schema_counts = Counter()
    ids = set()
    urls = set()
    row_count = 0
    empty_text = 0
    character_count = 0

    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("Archive contains a corrupted member")
        if member not in bundle.namelist():
            raise ValueError(f"Archive does not contain {member!r}")

        with bundle.open(member) as handle:
            for line_number, raw in enumerate(handle, start=1):
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at line {line_number}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Row {line_number} is not a JSON object")

                row_count += 1
                schema_counts[tuple(sorted(row))] += 1
                missing = set(REQUIRED_FIELDS) - row.keys()
                if missing:
                    missing_fields.update(missing)
                    continue

                if not row["_id"] or row["_id"] != row["id"]:
                    raise ValueError(f"Invalid or mismatched id at line {line_number}")
                if row["_id"] in ids:
                    raise ValueError(f"Duplicate id at line {line_number}: {row['_id']}")
                ids.add(row["_id"])

                parsed_url = urlparse(row["url"])
                if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
                    raise ValueError(f"Invalid URL at line {line_number}")
                urls.add(row["url"])

                text = row["text"]
                if not isinstance(text, str) or not text.strip():
                    empty_text += 1
                else:
                    character_count += len(text)

    if missing_fields:
        raise ValueError(f"Missing fields: {dict(missing_fields)}")
    if empty_text:
        raise ValueError(f"Rows with empty text: {empty_text}")

    return {
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256_file(archive),
        "member": member,
        "rows": row_count,
        "unique_ids": len(ids),
        "unique_urls": len(urls),
        "characters": character_count,
        "schema": [list(fields) for fields in sorted(schema_counts)],
        "required_fields": list(REQUIRED_FIELDS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the MTRAG Government corpus")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--member", default="govt.jsonl")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--expected-rows", type=int)
    args = parser.parse_args()

    report = validate_archive(args.input, args.member)
    if args.expected_sha256 and report["archive_sha256"].lower() != args.expected_sha256.lower():
        raise ValueError("Archive SHA-256 does not match the expected value")
    if args.expected_rows is not None and report["rows"] != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows} rows, got {report['rows']}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
