import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "projects/retail-repeat-purchase/src"))
from build_labels import build_labels  # noqa: E402


class BuildLabelsTest(unittest.TestCase):
    def test_half_open_windows_and_eligibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                [1, "099", "2020-01-02T00:00:00", 1, 10.0, "purchase"],  # covers observation start
                [1, "100", "2020-01-10T10:00:00", 1, 10.0, "purchase"],
                [1, "101", "2020-02-01T00:00:00", 1, 10.0, "purchase"],  # cutoff belongs to prediction
                [1, "102", "2020-02-10T00:00:00", 1, 10.0, "purchase"],  # prediction end is exclusive
                [2, "200", "2020-01-15T10:00:00", 1, 5.0, "purchase"],
                [3, "300", "2020-02-05T10:00:00", 1, 5.0, "purchase"],  # no observation history
                [4, "400", "2020-02-11T10:00:00", 1, 5.0, "purchase"],  # extends source coverage
            ]
            purchases = root / "purchases.csv"
            pd.DataFrame(rows, columns=["customer_id", "invoice_id", "invoice_date", "quantity", "line_revenue", "event_type"]).to_csv(purchases, index=False)
            output, summary_path = root / "labels.csv", root / "summary.json"
            summary = build_labels(purchases, output, summary_path, [pd.Timestamp("2020-02-01")], 30, 9)
            labels = pd.read_csv(output)
            self.assertEqual(labels[["customer_id", "label"]].to_dict("records"), [{"customer_id": 1, "label": 1}, {"customer_id": 2, "label": 0}])
            self.assertEqual(summary["label_counts"], {"0": 1, "1": 1})
            self.assertEqual(json.loads(summary_path.read_text())["output"]["rows"], 2)

    def test_cancellation_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            purchases = root / "purchases.csv"
            pd.DataFrame([[1, "C100", "2020-01-10", 1, 5.0, "cancellation"]], columns=["customer_id", "invoice_id", "invoice_date", "quantity", "line_revenue", "event_type"]).to_csv(purchases, index=False)
            with self.assertRaisesRegex(ValueError, "event_type=purchase"):
                build_labels(purchases, root / "labels.csv", root / "summary.json", [pd.Timestamp("2020-02-01")])


if __name__ == "__main__":
    unittest.main()
