import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "projects/retail-repeat-purchase/src"))
from tune_baselines import tune_models  # noqa: E402


class TuneBaselinesTest(unittest.TestCase):
    def _features(self):
        rows = []
        for customer_id in range(1, 61):
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

    def test_deterministic_cv_and_serialization(self):
        frame = self._features()
        artifact_one, report_one = tune_models(frame, random_state=7)
        artifact_two, report_two = tune_models(frame, random_state=7)
        self.assertEqual(report_one, report_two)
        self.assertEqual(report_one["cv_folds"], 3)
        self.assertEqual(report_one["train_rows"], 48)
        self.assertEqual(report_one["test_rows"], 12)
        self.assertEqual(set(report_one["models"]), {"logistic_regression", "random_forest"})
        self.assertEqual(report_one["baseline"]["model"], "dummy")
        for result in report_one["models"].values():
            self.assertLessEqual(result["configurations"], 20)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            joblib.dump(artifact_one, path)
            loaded = joblib.load(path)
            predictions = loaded["model"].predict(frame[loaded["feature_columns"]])
            self.assertEqual(len(predictions), len(frame))

    def test_rejects_configuration_limit(self):
        with self.assertRaisesRegex(ValueError, "configurations"):
            tune_models(self._features(), max_configs=5)

    def test_rejects_invalid_cv_folds(self):
        with self.assertRaisesRegex(ValueError, "cv_folds"):
            tune_models(self._features(), cv_folds=1)


if __name__ == "__main__":
    unittest.main()
