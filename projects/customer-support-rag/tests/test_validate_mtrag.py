import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from validate_mtrag import validate_archive  # noqa: E402


class ValidateMtragTest(unittest.TestCase):
    def _archive(self, rows):
        directory = tempfile.TemporaryDirectory()
        archive = Path(directory.name) / "fixture.zip"
        payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("govt.jsonl", payload)
        return directory, archive

    def test_valid_archive_reports_rows_and_unique_sources(self):
        directory, archive = self._archive(
            [
                {
                    "_id": "1",
                    "id": "1",
                    "url": "https://example.gov/a",
                    "title": "A",
                    "text": "First passage.",
                },
                {
                    "_id": "2",
                    "id": "2",
                    "url": "https://example.gov/a",
                    "title": "A",
                    "text": "Second passage.",
                },
            ]
        )
        self.addCleanup(directory.cleanup)
        report = validate_archive(archive)
        self.assertEqual(report["rows"], 2)
        self.assertEqual(report["unique_ids"], 2)
        self.assertEqual(report["unique_urls"], 1)

    def test_duplicate_ids_fail(self):
        directory, archive = self._archive(
            [
                {"_id": "1", "id": "1", "url": "https://example.gov/a", "title": "A", "text": "x"},
                {"_id": "1", "id": "1", "url": "https://example.gov/b", "title": "B", "text": "y"},
            ]
        )
        self.addCleanup(directory.cleanup)
        with self.assertRaisesRegex(ValueError, "Duplicate id"):
            validate_archive(archive)


if __name__ == "__main__":
    unittest.main()
