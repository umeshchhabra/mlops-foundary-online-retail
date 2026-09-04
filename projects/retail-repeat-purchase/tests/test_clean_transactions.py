import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "projects/retail-repeat-purchase/src"))
from clean_transactions import clean  # noqa: E402


class CleanTransactionsTest(unittest.TestCase):
    def test_policy_and_audit_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                ["100", "A", "item", 2, "2020-01-01 10:00:00", 5.0, 1, "UK"],
                ["100", "A", "item", 2, "2020-01-01 10:00:00", 5.0, 1, "UK"],  # exact duplicate
                ["101", "B", "item", 1, "2020-01-02 10:00:00", 4.0, None, "UK"],  # no customer
                ["C102", "C", "item", 1, "2020-01-03 10:00:00", 4.0, 1, "UK"],  # cancellation
                ["103", "D", "item", -1, "2020-01-04 10:00:00", 4.0, 1, "UK"],  # return
                ["104", "E", "item", 1, "2020-01-05 10:00:00", 0.0, 1, "UK"],  # invalid price
                ["105", "F", None, 3, "2020-01-06 10:00:00", 2.0, 2, "UK"],  # valid without description
                ["106", "G", "bad quantity", "not-a-number", "2020-01-07 10:00:00", 2.0, 2, "UK"],  # unparseable quantity
            ]
            input_path = root / "input.csv"
            pd.DataFrame(rows, columns=["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"]).to_csv(input_path, index=False)
            output_path, summary_path = root / "clean.csv", root / "summary.json"
            summary = clean(input_path, output_path, summary_path)
            self.assertEqual(summary["input"]["rows"], 8)
            self.assertEqual(summary["output"]["rows"], 2)
            self.assertEqual(summary["exclusions"], {"cancellation": 1, "exact_duplicate": 1, "invalid_quantity": 1, "missing_customer_id": 1, "nonpositive_quantity": 1, "nonpositive_unit_price": 1})
            with output_path.open(newline="") as output_file:
                output = list(csv.DictReader(output_file))
            self.assertEqual([row["invoice_id"] for row in output], ["100", "105"])
            self.assertEqual(output[0]["line_revenue"], "10.0")
            self.assertEqual(output[1]["line_revenue"], "6.0")
            self.assertEqual(json.loads(summary_path.read_text())["output"]["rows"], 2)

    def test_missing_schema_fails_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "bad.csv"
            pd.DataFrame({"InvoiceNo": [1]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Required columns"):
                clean(path, root / "out.csv", root / "summary.json")
            self.assertFalse((root / "out.csv").exists())


if __name__ == "__main__":
    unittest.main()
