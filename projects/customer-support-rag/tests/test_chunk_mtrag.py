import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chunk_mtrag import chunk_archive, split_text  # noqa: E402


class ChunkMtragTest(unittest.TestCase):
    def test_split_text_is_deterministic_and_bounded(self):
        text = " ".join(f"word{i}" for i in range(80))
        first = split_text(text, max_chars=80, overlap_chars=15)
        second = split_text(text, max_chars=80, overlap_chars=15)
        self.assertEqual(first, second)
        self.assertTrue(all(len(chunk) <= 80 for chunk in first))
        self.assertGreater(len(first), 1)

    def test_chunk_archive_preserves_document_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "input.zip"
            output = root / "chunks.jsonl"
            rows = [
                {
                    "_id": "doc-1",
                    "id": "doc-1",
                    "url": "https://example.gov/a",
                    "title": "A",
                    "text": "One two three four five six seven eight nine ten.",
                }
            ]
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("govt.jsonl", "\n".join(json.dumps(row) for row in rows) + "\n")

            report = chunk_archive(archive, output, max_chars=25, overlap_chars=5)
            chunks = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(report["documents"], 1)
            self.assertEqual(report["chunks"], len(chunks))
            self.assertTrue(all(chunk["document_id"] == "doc-1" for chunk in chunks))
            self.assertEqual(
                [chunk["chunk_index"] for chunk in chunks], list(range(len(chunks)))
            )
            self.assertTrue(all(len(chunk["text"]) <= 25 for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
