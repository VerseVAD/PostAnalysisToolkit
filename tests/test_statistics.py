from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import compare
import correlation
import robustness
import sensitivity


class StatisticalRegressionTests(unittest.TestCase):
    def test_cliffs_delta_direction_and_ties(self) -> None:
        self.assertEqual(compare.cliff_delta(np.array([3, 4]), np.array([1, 2])), 1.0)
        self.assertEqual(compare.cliff_delta(np.array([1, 2]), np.array([3, 4])), -1.0)
        self.assertEqual(compare.cliff_delta(np.array([1, 2]), np.array([1, 2])), 0.0)

    def test_paired_bootstrap_is_fixed_seed_reproducible(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.2, 1.7, 3.4, 3.8, 5.1])
        first = correlation.paired_bootstrap_ci(
            x, y, method="spearman", n_resamples=500, seed=12345
        )
        second = correlation.paired_bootstrap_ci(
            x, y, method="spearman", n_resamples=500, seed=12345
        )
        self.assertEqual(first, second)
        # Degenerate resamples are intentionally omitted from percentile bounds.
        self.assertGreater(first[2], 0)
        self.assertLessEqual(first[2], 500)

    def test_leave_one_out_runs_once_per_work(self) -> None:
        frame = pd.DataFrame(
            {
                "text_id": ["a", "b", "c", "d", "e"],
                "title": ["A", "B", "C", "D", "E"],
                "x_value": [1, 2, 3, 4, 5],
                "y_value": [2, 1, 4, 3, 5],
            }
        )
        summary, details = robustness.correlation_leave_one_out(frame)
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["leave_one_out_runs"], 5)
        self.assertEqual(len(details), 5)

    def test_common_set_uses_same_works_for_each_variant(self) -> None:
        variants = [
            sensitivity.Variant("m|d|", "Metric", "r", "Resource", "m", "d", "", "all_matched", "token", "0-1"),
            sensitivity.Variant("m|d|", "Metric", "r", "Resource", "m", "d", "", "content_words", "token", "0-1"),
        ]
        rows = []
        for text_id, values in {"a": (1.0, 1.1), "b": (2.0, 2.1), "c": (3.0, None)}.items():
            for variant, value in zip(variants, values):
                rows.append(
                    {
                        "metric_key": variant.metric_key,
                        "variant_id": variant.variant_id,
                        "text_id": text_id,
                        "title": text_id.upper(),
                        "author": "",
                        "collection": "",
                        "date_label": "",
                        "genre": "poem",
                        "value": value,
                        "coverage": 1.0 if value is not None else np.nan,
                        "observations": 10 if value is not None else 0,
                    }
                )
        result = sensitivity.analyze_metric(
            "m|d|", "Metric", pd.DataFrame(rows), variants, None, "common", 3
        )
        self.assertEqual(result["summary"]["common_qualifying_n"], 2)
        self.assertTrue((result["corpus_variants"]["analysis_n"] == 2).all())


if __name__ == "__main__":
    unittest.main()
