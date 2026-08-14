from __future__ import annotations

import sys
import tempfile
import unittest
import csv
import io
import zipfile
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from versevad_reader import BASELINE_COLUMNS, MetricSpec, VerseVADCorpusReader
from versevad_tools.audit import AuditSourceError, require_audit


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

    def test_wrong_mode_and_current_view_are_rejected_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            single_path = Path(temporary) / "single.zip"
            with zipfile.ZipFile(single_path, "w") as archive:
                archive.writestr(
                    "03_MASTER_DATA/Master_Metrics.csv",
                    "export_schema_version,analysis_mode,metric_id,work_id,value\n",
                )
                archive.writestr(
                    "05_REPRODUCIBILITY/Export_Metadata.csv",
                    "export_schema_version,analysis_mode,export_type\n"
                    "3.0,single_poem,complete_audit\n",
                )
            with self.assertRaisesRegex(AuditSourceError, "requires a Corpus"):
                require_audit(
                    single_path,
                    expected_analysis_mode="corpus",
                    require_complete=True,
                )

            current_path = Path(temporary) / "current.zip"
            with zipfile.ZipFile(current_path, "w") as archive:
                archive.writestr(
                    "03_MASTER_DATA/Master_Metrics.csv",
                    "export_schema_version,analysis_mode,metric_id,work_id,value\n",
                )
                archive.writestr(
                    "05_REPRODUCIBILITY/Export_Metadata.csv",
                    "export_schema_version,analysis_mode,export_type\n"
                    "3.0,corpus,current_view\n",
                )
            with self.assertRaisesRegex(AuditSourceError, "requires a Complete Audit"):
                require_audit(
                    current_path,
                    expected_analysis_mode="corpus",
                    require_complete=True,
                )

    def test_reads_schema_v3_corpus_audit_and_accepts_canonical_metric_id(self) -> None:
        fields = (
            "export_schema_version", "analysis_id", "analysis_mode", "work_id",
            "title", "author", "collection", "date_label", "genre", "module_id",
            "metric_id", "metric_label", "legacy_metric_id", "dimension", "category",
            "resource_id", "resource_label", "resource_version", "lexical_scope",
            "weighting", "analysis_level", "corpus_aggregation", "value", "unit",
            "denominator", "eligible_token_count", "matched_token_count",
            "unmatched_token_count", "token_coverage", "eligible_type_count",
            "matched_type_count", "unmatched_type_count", "type_coverage",
            "observation_count", "notes",
        )
        row = {field: "" for field in fields}
        row.update(
            {
                "export_schema_version": "3.0",
                "analysis_id": "run",
                "analysis_mode": "corpus",
                "work_id": "poem-1",
                "title": "Synthetic poem",
                "module_id": "vad",
                "metric_id": "vad.valence.mean",
                "legacy_metric_id": "vad_mean",
                "dimension": "valence",
                "resource_id": "nrc_vad_v2_1",
                "resource_label": "NRC VAD Lexicon v2.1",
                "lexical_scope": "stopword_excluded",
                "weighting": "token",
                "analysis_level": "work",
                "value": "0.6",
                "unit": "normalized 0-1",
                "eligible_token_count": "12",
                "matched_token_count": "10",
                "unmatched_token_count": "2",
                "token_coverage": str(10 / 12),
                "observation_count": "10",
            }
        )
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
        metadata = (
            "export_schema_version,analysis_mode,export_type\n"
            "3.0,corpus,complete_audit\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "corpus_complete_audit.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(
                    "03_MASTER_DATA/Master_Metrics.csv",
                    "\ufeff" + output.getvalue(),
                )
                archive.writestr(
                    "05_REPRODUCIBILITY/Export_Metadata.csv",
                    metadata,
                )
            reader = VerseVADCorpusReader(path)
            report = reader.validate()
            self.assertEqual(report.export_schema_version, "3.0")
            selected = reader.select_metric(
                MetricSpec(
                    lexicon_id="nrc_vad_v2_1",
                    metric="vad.valence.mean",
                    dimension="valence",
                    analysis_view="stopwords_excluded",
                    weighting="token",
                )
            )
            self.assertEqual(len(selected), 1)
            self.assertAlmostEqual(float(selected.iloc[0]["value"]), 0.6)


if __name__ == "__main__":
    unittest.main()
