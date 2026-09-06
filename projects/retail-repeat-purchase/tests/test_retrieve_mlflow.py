import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "projects/retail-repeat-purchase/src"))
from package_model import sha256  # noqa: E402
from retrieve_mlflow import verify_package  # noqa: E402


class RetrieveMlflowTest(unittest.TestCase):
    def _files(self, directory: Path):
        frame = pd.DataFrame({"feature_a": [0, 1, 2, 3], "feature_b": [1.0, 1.5, 3.0, 4.0]})
        labels = np.array([0, 0, 1, 1])
        model = LogisticRegression(random_state=42).fit(frame, labels)
        schema = {
            "feature_a": {"dtype": "int64", "kind": "numeric", "nullable": False},
            "feature_b": {"dtype": "float64", "kind": "numeric", "nullable": False},
        }
        package = {"model": model, "feature_columns": list(frame.columns), "input_schema": schema}
        local_model = directory / "local.joblib"
        downloaded_model = directory / "downloaded.joblib"
        features = directory / "features.csv"
        report = directory / "tuning-report.json"
        manifest = directory / "model-package.json"
        joblib.dump(package, local_model)
        shutil.copyfile(local_model, downloaded_model)
        frame.to_csv(features, index=False)
        report.write_text("{}", encoding="utf-8")
        predictions = model.predict(frame)
        encoded = json.dumps(predictions.tolist(), separators=(",", ":")).encode("utf-8")
        import hashlib
        manifest.write_text(json.dumps({
            "model": {"sha256": sha256(local_model)},
            "dataset_sha256": sha256(features),
            "report_sha256": sha256(report),
            "input_schema": schema,
            "prediction_check": {"prediction_sha256": hashlib.sha256(encoded).hexdigest()},
        }), encoding="utf-8")
        return local_model, downloaded_model, manifest, report, features

    def test_verifies_hash_schema_and_prediction_parity(self):
        with tempfile.TemporaryDirectory() as directory_name:
            files = self._files(Path(directory_name))
            result = verify_package(*files)
            self.assertTrue(result["prediction_parity"])
            self.assertTrue(result["schema_verified"])
            self.assertEqual(result["rows"], 4)

    def test_rejects_changed_downloaded_artifact(self):
        with tempfile.TemporaryDirectory() as directory_name:
            files = self._files(Path(directory_name))
            files[1].write_bytes(files[1].read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "Downloaded model hash"):
                verify_package(*files)


if __name__ == "__main__":
    unittest.main()
