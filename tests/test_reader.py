from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from versevad_reader import BASELINE_COLUMNS, MetricSpec, VerseVADCorpusReader


class ReaderSchemaTests(unittest.TestCase):
    def test_validate_and_extract_exact_metric(self) -> None:
        row = {column: "" for column in BASELINE_COLUMNS}
        row.update(
            {
                "run_id": "run",
                "text_id": "poem-1",
                "text_version_id": "version-1",
                "title": "Synthetic poem",
                "author": "Test",
                "genre": "poem",
                "lexicon_id": "nrc_vad_v2_1",
                "lexicon": "NRC VAD Lexicon v2.1",
                "value_kind": "continuous",
                "metric": "vad_mean",
                "dimension": "valence",
                "weighting": "token",
                "scale": "normalized 0-1",
                "denominator": "matched observations",
                "value": 0.6,
                "observations": 10,
                "matched_tokens": 10,
                "lexical_tokens": 12,
                "coverage": 10 / 12,
                "completed_at": "2026-01-01T00:00:00Z",
                "analysis_view": "stopwords_excluded",
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "corpus_vad_metrics.csv"
            pd.DataFrame([row], columns=BASELINE_COLUMNS).to_csv(path, index=False)
            reader = VerseVADCorpusReader(path)
            report = reader.validate()
            self.assertEqual(report.work_count, 1)
            selected = reader.select_metric(
                MetricSpec(
                    lexicon_id="nrc_vad_v2_1",
                    metric="vad_mean",
                    dimension="valence",
                    analysis_view="stopwords_excluded",
                    weighting="token",
                )
            )
            self.assertEqual(len(selected), 1)
            self.assertAlmostEqual(float(selected.iloc[0]["value"]), 0.6)


if __name__ == "__main__":
    unittest.main()
