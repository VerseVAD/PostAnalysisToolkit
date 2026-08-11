from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from versevad_tools.sources import discover_corpus_metric_sources, discover_files


class SourceDiscoveryTests(unittest.TestCase):
    def test_generic_and_validated_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "corpus_vad_metrics.csv").write_text("header\n", encoding="utf-8")
            (root / "other.csv").write_text("header\n", encoding="utf-8")
            with zipfile.ZipFile(root / "valid.zip", "w") as archive:
                archive.writestr("nested/corpus_vad_metrics.csv", "header\n")
            with zipfile.ZipFile(root / "unrelated.zip", "w") as archive:
                archive.writestr("notes.txt", "not an audit")
            (root / "broken.zip").write_bytes(b"not a zip")

            self.assertEqual(len(discover_files(root)), 5)
            compatible = discover_corpus_metric_sources(root, "corpus_vad_metrics.csv")
            self.assertEqual([path.name for path in compatible], ["corpus_vad_metrics.csv", "valid.zip"])


if __name__ == "__main__":
    unittest.main()
