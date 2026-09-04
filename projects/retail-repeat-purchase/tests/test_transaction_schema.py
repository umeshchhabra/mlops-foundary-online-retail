import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "pipelines/shared"))
from transaction_schema import DataValidationError, read_transactions  # noqa: E402


class TransactionSchemaTest(unittest.TestCase):
    COLUMNS = ["customer_id", "invoice_id", "stock_code", "invoice_date", "quantity", "line_revenue", "event_type"]

    def test_normalizes_valid_events_and_keeps_negative_quantities(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.csv"
            pd.DataFrame([["1", "100", "A", "2020-01-01", "-2", "-10", "cancellation"]], columns=self.COLUMNS).to_csv(path, index=False)
            result = read_transactions(path, name="events", view="events")
            self.assertEqual(result.customer_id.dtype, "int64")
            self.assertEqual(result.loc[0, "event_type"], "cancellation")
            self.assertEqual(result.loc[0, "quantity"], -2)

    def test_reports_schema_and_parse_errors_together(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.csv"
            pd.DataFrame([["bad", "", "A", "not-a-date", "x", 1, "mystery"]], columns=self.COLUMNS).to_csv(path, index=False)
            with self.assertRaisesRegex(DataValidationError, "customer_id.*unparseable.*quantity.*unparseable.*invoice_date.*unparseable.*event_type: unknown values"):
                read_transactions(path, name="events", view="events")

    def test_purchase_view_rejects_non_purchase_events(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "purchases.csv"
            pd.DataFrame([[1, "C100", "A", "2020-01-01", 1, 5, "cancellation"]], columns=self.COLUMNS).to_csv(path, index=False)
            with self.assertRaisesRegex(DataValidationError, "event_type must be purchase"):
                read_transactions(path, name="purchases", view="purchases")


if __name__ == "__main__":
    unittest.main()
