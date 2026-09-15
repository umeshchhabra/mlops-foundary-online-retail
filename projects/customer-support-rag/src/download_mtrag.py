import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


DEFAULT_URL = (
    "https://github.com/IBM/mt-rag-benchmark/raw/2c618bb98db3c8526433e22d8a2f7320f10a7470/"
    "corpora/passage_level/govt.jsonl.zip"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, output: Path, expected_sha256: str | None = None) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".part")
    if partial.exists():
        partial.unlink()

    with urllib.request.urlopen(url) as response, partial.open("wb") as handle:
        while block := response.read(1024 * 1024):
            handle.write(block)

    checksum = sha256_file(partial)
    if expected_sha256 and checksum.lower() != expected_sha256.lower():
        partial.unlink()
        raise ValueError(
            f"SHA-256 mismatch: expected {expected_sha256}, got {checksum}"
        )

    partial.replace(output)
    return {
        "source_url": url,
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": checksum,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the MTRAG Government corpus")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--sha256")
    args = parser.parse_args()
    print(json.dumps(download(args.url, args.output, args.sha256), indent=2))


if __name__ == "__main__":
    main()
