import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "projects/retail-repeat-purchase/src"))
from track_mlflow import build_run_payload  # noqa: E402


class TrackMlflowTest(unittest.TestCase):
    def test_payload_is_safe_and_contains_lineage(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            report = {
                "selected_model": "logistic_regression",
                "random_state": 42,
                "test_size": 0.2,
                "cv_folds": 3,
                "scoring": "f1_macro",
                "feature_columns": ["a", "b"],
                "baseline": {"test": {"macro_f1": 0.3}},
                "models": {"logistic_regression": {
                    "best_params": {"classifier__C": 0.1, "classifier__class_weight": None},
                    "cv_macro_f1": 0.6,
                    "cv_macro_f1_std": 0.01,
                    "test": {"macro_f1": 0.65, "accuracy": 0.7},
                }},
            }
            manifest = {
                "dataset_sha256": "data-hash",
                "report_sha256": "report-hash",
                "model": {"sha256": "model-hash"},
            }
            model = directory / "model.joblib"
            report_path = directory / "report.json"
            manifest_path = directory / "manifest.json"
            payload = build_run_payload(report, manifest, model, report_path, manifest_path)
            self.assertEqual(payload["params"]["selected_classifier__class_weight"], "None")
            self.assertEqual(payload["metrics"]["selected_test_macro_f1"], 0.65)
            self.assertEqual(payload["tags"]["dataset_sha256"], "data-hash")
            self.assertEqual(payload["artifacts"][0][1], "model")


if __name__ == "__main__":
    unittest.main()
