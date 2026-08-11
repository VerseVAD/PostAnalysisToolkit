from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from versevad_tools.cli import parse_coverage_threshold, parse_index_selection


class SelectionParserTests(unittest.TestCase):
    def test_documented_selection_forms(self) -> None:
        self.assertEqual(parse_index_selection("A", 6), [0, 1, 2, 3, 4, 5])
        self.assertEqual(parse_index_selection("1", 6), [0])
        self.assertEqual(parse_index_selection("1,2", 6), [0, 1])
        self.assertEqual(parse_index_selection("1, 3, 5", 6), [0, 2, 4])
        self.assertEqual(parse_index_selection("5-6", 6), [4, 5])
        self.assertEqual(parse_index_selection("1,3,5-6", 6), [0, 2, 4, 5])

    def test_count_and_range_validation(self) -> None:
        with self.assertRaises(ValueError):
            parse_index_selection("1", 6, min_count=2)
        with self.assertRaises(ValueError):
            parse_index_selection("1-6", 6, max_count=5)
        with self.assertRaises(ValueError):
            parse_index_selection("7", 6)
        with self.assertRaises(ValueError):
            parse_index_selection("A", 6, allow_all=False)

    def test_one_based_compatibility(self) -> None:
        self.assertEqual(parse_index_selection("1,3-4", 5, one_based=True), [1, 3, 4])

    def test_coverage_parser(self) -> None:
        self.assertAlmostEqual(parse_coverage_threshold("80"), 0.8)
        self.assertAlmostEqual(parse_coverage_threshold("80%"), 0.8)
        self.assertAlmostEqual(parse_coverage_threshold("0.80"), 0.8)
        self.assertIsNone(parse_coverage_threshold(""))
        self.assertEqual(parse_coverage_threshold("", blank=0.0), 0.0)
        with self.assertRaises(ValueError):
            parse_coverage_threshold("120")


if __name__ == "__main__":
    unittest.main()
