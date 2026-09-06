import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "projects/retail-repeat-purchase/src"))
from train_baselines import train_baselines  # noqa: E402


class TrainBaselinesTest(unittest.TestCase):
    def _features(self):
        rows = []
        for customer_id in range(1, 41):
            label = customer_id % 2
            rows.append({
                "customer_id": customer_id,
                "cutoff_date": "2020-02-01T00:00:00",
                "observation_start": "2020-01-01T00:00:00",
                "prediction_end": "2020-03-01T00:00:00",
                "label": label,
                "purchase_count": 10 + label * 5,
                "cancellation_rate": 0.1 * label,
            })
        return pd.DataFrame(rows)

    def test_deterministic_comparison_and_artifact(self):
        frame = self._features()
        artifact_one, report_one = train_baselines(frame, random_state=7)
        artifact_two, report_two = train_baselines(frame, random_state=7)
        self.assertEqual(report_one, report_two)
        self.assertEqual(artifact_one["selected_model"], artifact_two["selected_model"])
        self.assertEqual(report_one["train_rows"], 32)
        self.assertEqual(report_one["test_rows"], 8)
        self.assertEqual(set(report_one["models"]), {"dummy", "logistic_regression", "random_forest"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            joblib.dump(artifact_one, path)
            loaded = joblib.load(path)
            predictions = loaded["model"].predict(frame[loaded["feature_columns"]])
            self.assertEqual(len(predictions), len(frame))

    def test_requires_both_classes(self):
        frame = self._features()
        frame["label"] = 1
        with self.assertRaisesRegex(ValueError, "both label classes"):
            train_baselines(frame)

    def test_rejects_non_numeric_feature(self):
        frame = self._features()
        frame["purchase_count"] = "unknown"
        with self.assertRaisesRegex(ValueError, "non-numeric"):
            train_baselines(frame)


if __name__ == "__main__":
    unittest.main()
