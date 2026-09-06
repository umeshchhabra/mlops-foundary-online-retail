import json
import sys
import tempfile
import unittest
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "projects/retail-repeat-purchase/src"))
from package_model import package_model, sha256  # noqa: E402
from tune_baselines import tune_models  # noqa: E402


class PackageModelTest(unittest.TestCase):
    def _features(self):
        import pandas as pd

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

    def _inputs(self, directory: Path):
        frame = self._features()
        artifact, report = tune_models(frame, random_state=7)
        model = directory / "tuned.joblib"
        report_path = directory / "tuning.json"
        features = directory / "features.csv"
        code = directory / "train.py"
        joblib.dump(artifact, model)
        features.write_text(frame.to_csv(index=False), encoding="utf-8")
        report["input_sha256"] = sha256(features)
        report_path.write_text(json.dumps(report), encoding="utf-8")
        code.write_text("training code", encoding="utf-8")
        return model, report_path, features, code

    def test_packages_schema_lineage_and_predictions(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            model, report, features, code = self._inputs(directory)
            output = directory / "package.joblib"
            manifest_path = directory / "manifest.json"
            manifest = package_model(model, report, features, code, output, manifest_path)
            package = joblib.load(output)
            self.assertEqual(package["input_schema"]["purchase_count"]["kind"], "numeric")
            self.assertEqual(manifest["prediction_check"]["rows"], 60)
            self.assertEqual(manifest["selected_model"], package["selected_model"])
            self.assertTrue(manifest["evaluation"]["selected_test_macro_f1"] > manifest["evaluation"]["baseline_macro_f1"])

    def test_rejects_report_without_baseline_improvement(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            model, report_path, features, code = self._inputs(directory)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["baseline"]["test"]["macro_f1"] = 1.0
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not improve"):
                package_model(model, report_path, features, code, directory / "package.joblib", directory / "manifest.json")


if __name__ == "__main__":
    unittest.main()
