import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_h3_resolution_ablation import split_panel


class H3ResolutionAblationSplitTest(unittest.TestCase):
    def test_uses_date_boundary_for_last_week_test_set(self):
        slots = pd.date_range("2014-06-01", "2014-06-30 23:00:00", freq="1h")
        panel = pd.DataFrame({"time_slot": slots, "order_count": 1.0})
        args = SimpleNamespace(train_ratio=0.7, valid_ratio=0.15, test_start_date="2014-06-24")

        train_df, valid_df, test_df = split_panel(panel, args)

        self.assertEqual(test_df["time_slot"].min(), pd.Timestamp("2014-06-24"))
        self.assertEqual(test_df["time_slot"].max(), pd.Timestamp("2014-06-30 23:00:00"))
        self.assertEqual(test_df["time_slot"].nunique(), 168)
        self.assertLess(train_df["time_slot"].max(), pd.Timestamp("2014-06-24"))
        self.assertLess(valid_df["time_slot"].max(), pd.Timestamp("2014-06-24"))


if __name__ == "__main__":
    unittest.main()
