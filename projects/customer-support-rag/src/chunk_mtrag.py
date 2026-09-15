import argparse
import json
import re
from pathlib import Path

from validate_mtrag import sha256_file


WHITESPACE = re.compile(r"\s+")


def _words(text: str) -> list[str]:
    return WHITESPACE.sub(" ", text).strip().split(" ")


def split_text(text: str, max_chars: int = 1200, overlap_chars: int = 200) -> list[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be non-negative and smaller than max_chars")

    words = _words(text)
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start
        length = 0
        while end < len(words):
            word_length = len(words[end])
            candidate_length = word_length if end == start else length + 1 + word_length
            if end > start and candidate_length > max_chars:
                break
            length = candidate_length
            end += 1

        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break

        overlap = 0
        next_start = end
        while next_start > start and overlap + len(words[next_start - 1]) + 1 <= overlap_chars:
            next_start -= 1
            overlap += len(words[next_start]) + 1
        start = max(start + 1, next_start)

    return chunks


def chunk_archive(
    archive: Path,
    output: Path,
    member: str = "govt.jsonl",
    max_chars: int = 1200,
    overlap_chars: int = 200,
) -> dict:
    if not archive.is_file():
        raise ValueError(f"Archive does not exist: {archive}")

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".part")
    if partial.exists():
        partial.unlink()

    documents = 0
    chunks = 0
    with __import__("zipfile").ZipFile(archive) as bundle, bundle.open(member) as handle:
        with partial.open("w", encoding="utf-8", newline="\n") as destination:
            for raw in handle:
                row = json.loads(raw)
                document_chunks = split_text(row["text"], max_chars, overlap_chars)
                documents += 1
                for index, text in enumerate(document_chunks):
                    chunk = {
                        "chunk_id": f"{row['_id']}#chunk-{index:04d}",
                        "document_id": row["_id"],
                        "chunk_index": index,
                        "chunk_count": len(document_chunks),
                        "title": row["title"],
                        "url": row["url"],
                        "text": text,
                    }
                    destination.write(
                        json.dumps(chunk, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    chunks += 1

    partial.replace(output)
    return {
        "input_archive": str(archive),
        "input_sha256": sha256_file(archive),
        "input_member": member,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "documents": documents,
        "chunks": chunks,
        "max_chars": max_chars,
        "overlap_chars": overlap_chars,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic MTRAG document chunks")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--member", default="govt.jsonl")
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--overlap-chars", type=int, default=200)
    args = parser.parse_args()
    print(
        json.dumps(
            chunk_archive(
                args.input,
                args.output,
                args.member,
                args.max_chars,
                args.overlap_chars,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
