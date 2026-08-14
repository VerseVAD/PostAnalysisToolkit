"""Shared schema detection and file access for VerseVAD Complete Audits."""

from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


MASTER_METRICS_PATH = "03_MASTER_DATA/Master_Metrics.csv"
EXPORT_METADATA_PATH = "05_REPRODUCIBILITY/Export_Metadata.csv"
LEGACY_CORPUS_BASENAME = "corpus_vad_metrics.csv"


class AuditSourceError(RuntimeError):
    """Raised when a file is not a compatible VerseVAD audit source."""


@dataclass(frozen=True)
class AuditDescriptor:
    schema_version: str
    analysis_mode: str
    export_type: str
    master_member: str
    legacy: bool


def _one_by_basename(names: list[str], basename: str) -> Optional[str]:
    matches = [name for name in names if Path(name).name.casefold() == basename.casefold()]
    if len(matches) > 1:
        raise AuditSourceError(
            f"The audit contains more than one {basename}; refusing to choose arbitrarily."
        )
    return matches[0] if matches else None


def describe_audit(path: str | Path) -> AuditDescriptor:
    source = Path(path).expanduser().resolve()
    if source.suffix.casefold() != ".zip":
        raise AuditSourceError("Expected a VerseVAD Complete Audit ZIP.")
    try:
        with zipfile.ZipFile(source) as archive:
            names = [name for name in archive.namelist() if not name.endswith("/")]
            metadata_member = (
                EXPORT_METADATA_PATH
                if EXPORT_METADATA_PATH in names
                else _one_by_basename(names, "Export_Metadata.csv")
            )
            master = (
                MASTER_METRICS_PATH
                if MASTER_METRICS_PATH in names
                else _one_by_basename(names, "Master_Metrics.csv")
            )
            if metadata_member and master:
                text = archive.read(metadata_member).decode("utf-8-sig")
                rows = list(csv.DictReader(io.StringIO(text)))
                if not rows:
                    raise AuditSourceError("Export_Metadata.csv is empty.")
                row = rows[0]
                return AuditDescriptor(
                    schema_version=str(row.get("export_schema_version", "")).strip(),
                    analysis_mode=str(row.get("analysis_mode", "")).strip(),
                    export_type=str(row.get("export_type", "")).strip(),
                    master_member=master,
                    legacy=False,
                )

            legacy_corpus = _one_by_basename(names, LEGACY_CORPUS_BASENAME)
            if legacy_corpus:
                return AuditDescriptor(
                    schema_version="legacy-corpus",
                    analysis_mode="corpus",
                    export_type="complete_audit",
                    master_member=legacy_corpus,
                    legacy=True,
                )
            legacy_single = _one_by_basename(names, "profile_comparison.csv")
            if legacy_single:
                return AuditDescriptor(
                    schema_version="legacy-single",
                    analysis_mode="single_poem",
                    export_type="complete_audit",
                    master_member=legacy_single,
                    legacy=True,
                )
    except zipfile.BadZipFile as exc:
        raise AuditSourceError(f"Not a readable ZIP archive: {source}") from exc
    raise AuditSourceError(
        "The ZIP does not contain a supported VerseVAD Complete Audit schema."
    )


def require_audit(
    path: str | Path,
    *,
    expected_analysis_mode: str,
    require_complete: bool = True,
) -> AuditDescriptor:
    descriptor = describe_audit(path)
    if descriptor.analysis_mode != expected_analysis_mode:
        friendly = {
            "single_poem": "Single Poem",
            "compare_poems": "Compare Poems",
            "corpus": "Corpus / Research Project",
            "other_text": "Other Text / Prose",
        }
        raise AuditSourceError(
            f"This tool requires a {friendly.get(expected_analysis_mode, expected_analysis_mode)} "
            f"Complete Audit. The selected ZIP is a "
            f"{friendly.get(descriptor.analysis_mode, descriptor.analysis_mode)} export."
        )
    if require_complete and descriptor.export_type != "complete_audit":
        raise AuditSourceError(
            "This tool requires a Complete Audit ZIP. The selected ZIP is a Current View export."
        )
    return descriptor


def resolve_member(names: set[str], legacy_name: str) -> Optional[str]:
    """Resolve one historical Single-Poem member in legacy or schema-v3 geometry."""

    aliases = {
        "00_START_HERE/profile_comparison.csv": "03_MASTER_DATA/All_Profiles.csv",
        "00_START_HERE/metric_dictionary.csv": (
            "05_REPRODUCIBILITY/Legacy_Metric_Dictionary.csv"
        ),
        "00_START_HERE/coverage_summary.csv": (
            "04_AUDIT/00_START_HERE/coverage_summary.csv"
        ),
        "07_PROCESSING_AUDIT/source.csv": "04_AUDIT/07_PROCESSING_AUDIT/source.csv",
        "07_PROCESSING_AUDIT/tokens.csv": "04_AUDIT/07_PROCESSING_AUDIT/tokens.csv",
    }
    candidates = [legacy_name, aliases.get(legacy_name, ""), f"04_AUDIT/{legacy_name}"]
    for candidate in candidates:
        if candidate and candidate in names:
            return candidate
    basename = Path(legacy_name).name.casefold()
    matches = [name for name in names if Path(name).name.casefold() == basename]
    return matches[0] if len(matches) == 1 else None


__all__ = [
    "AuditDescriptor",
    "AuditSourceError",
    "EXPORT_METADATA_PATH",
    "LEGACY_CORPUS_BASENAME",
    "MASTER_METRICS_PATH",
    "describe_audit",
    "require_audit",
    "resolve_member",
]
