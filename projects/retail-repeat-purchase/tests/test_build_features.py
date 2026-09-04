import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "projects/retail-repeat-purchase/src"))
from build_features import build_features  # noqa: E402


class BuildFeaturesTest(unittest.TestCase):
    def test_observation_only_features_and_event_signals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = pd.DataFrame([
                [1, "100", "A", "2020-01-10", 2, 10.0, "purchase"],
                [1, "C101", "A", "2020-01-20", -1, -5.0, "cancellation"],
                [1, "102", "B", "2020-02-05", 1, 7.0, "purchase"],  # future: must not affect features
                [2, "200", "C", "2020-01-15", 1, 3.0, "purchase"],
            ], columns=["customer_id", "invoice_id", "stock_code", "invoice_date", "quantity", "line_revenue", "event_type"])
            purchases = events.loc[events["event_type"].eq("purchase")].copy()
            labels = pd.DataFrame([[1, "2020-02-01", "2020-01-02", "2020-02-11", 1], [2, "2020-02-01", "2020-01-02", "2020-02-11", 0]], columns=["customer_id", "cutoff_date", "observation_start", "prediction_end", "label"])
            events_path, purchases_path, labels_path = root / "events.csv", root / "purchases.csv", root / "labels.csv"
            events.to_csv(events_path, index=False); purchases.to_csv(purchases_path, index=False); labels.to_csv(labels_path, index=False)
            output, summary_path = root / "features.csv", root / "summary.json"
            summary = build_features(events_path, purchases_path, labels_path, output, summary_path)
            result = pd.read_csv(output).set_index("customer_id")
            self.assertEqual(int(result.loc[1, "purchase_line_count"]), 1)
            self.assertEqual(int(result.loc[1, "event_count"]), 2)
            self.assertEqual(int(result.loc[1, "cancellation_count"]), 1)
            self.assertAlmostEqual(result.loc[1, "cancellation_rate"], 0.5)
            self.assertEqual(int(result.loc[1, "distinct_stock_code_count"]), 1)
            self.assertEqual(int(result.loc[2, "event_count"]), 1)
            self.assertEqual(summary["output"]["rows"], 2)
            self.assertEqual(json.loads(summary_path.read_text())["feature_nulls"]["event_count"], 0)

    def test_missing_label_column_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = pd.DataFrame([[1, "100", "A", "2020-01-10", 1, 2.0, "purchase"]], columns=["customer_id", "invoice_id", "stock_code", "invoice_date", "quantity", "line_revenue", "event_type"])
            valid.to_csv(root / "events.csv", index=False); valid.to_csv(root / "purchases.csv", index=False)
            pd.DataFrame({"customer_id": [1]}).to_csv(root / "labels.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Required label columns"):
                build_features(root / "events.csv", root / "purchases.csv", root / "labels.csv", root / "features.csv", root / "summary.json")


if __name__ == "__main__":
    unittest.main()
