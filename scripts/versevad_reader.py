#!/usr/bin/env python3
"""versevad_reader.py

A small, strict reader for VerseVAD corpus research exports.

Purpose
-------
This module is the import layer for a private downstream VerseVAD research
suite. It reads ONLY ``corpus_vad_metrics.csv`` (either directly or from a
VerseVAD Complete Audit ZIP), validates the expected schema, and returns clean
poem-level metric tables for later tools such as correlation, anomaly,
sensitivity, robustness, and corpus-comparison analyses.

It deliberately performs NO inferential statistics.

Current baseline
----------------
Built against the VerseVAD 1.0.0 corpus Complete Audit export schema observed
2026-08-11. Extra columns are tolerated by default; required columns may not be
missing.

Examples
--------
Smoke-test an audit ZIP::

    python versevad_reader.py source\\dickinson_complete_audit.zip

Show the metric catalog::

    python versevad_reader.py source\\dickinson_complete_audit.zip --catalog

Export one exact poem-level metric::

    python versevad_reader.py source\\dickinson_complete_audit.zip \\
        --lexicon-id brysbaert-concreteness-2014 \\
        --metric concreteness_concreteness_mean_mean \\
        --dimension concreteness_mean \\
        --analysis-view content_words \\
        --weighting token \\
        --output exports\\concreteness.csv

Programmatic use::

    from versevad_reader import VerseVADCorpusReader, MetricSpec

    reader = VerseVADCorpusReader("source/dickinson_complete_audit.zip")

    concreteness = MetricSpec(
        lexicon_id="brysbaert-concreteness-2014",
        metric="concreteness_concreteness_mean_mean",
        dimension="concreteness_mean",
        analysis_view="content_words",
        weighting="token",
    )

    interoception = MetricSpec(
        lexicon_id="lancaster-sensorimotor-2020",
        metric="sensorimotor_interoceptive_mean",
        dimension="interoceptive",
        analysis_view="content_words",
        weighting="token",
    )

    paired = reader.pair_metrics(concreteness, interoception)

Dependencies
------------
Python 3.10+ and pandas.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Optional, Sequence

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - friendly CLI failure
    raise SystemExit(
        "versevad_reader.py requires pandas. Install it with: python -m pip install pandas"
    ) from exc

from versevad_tools.core import configure_console_encoding


__version__ = "0.1.0"
METRICS_FILENAME = "corpus_vad_metrics.csv"

# Baseline schema from the stable VerseVAD Complete Audit export.
BASELINE_COLUMNS: tuple[str, ...] = (
    "run_id",
    "text_id",
    "text_version_id",
    "title",
    "author",
    "collection",
    "date_label",
    "genre",
    "lexicon_id",
    "lexicon",
    "value_kind",
    "metric",
    "dimension",
    "category",
    "weighting",
    "scale",
    "denominator",
    "value",
    "observations",
    "matched_tokens",
    "lexical_tokens",
    "coverage",
    "completed_at",
    "analysis_view",
)

# Every one of these is required for safe downstream metric selection.
REQUIRED_COLUMNS: frozenset[str] = frozenset(BASELINE_COLUMNS)

METADATA_COLUMNS: tuple[str, ...] = (
    "text_id",
    "text_version_id",
    "title",
    "author",
    "collection",
    "date_label",
    "genre",
)

NUMERIC_COLUMNS: tuple[str, ...] = (
    "value",
    "observations",
    "matched_tokens",
    "lexical_tokens",
    "coverage",
)

SUPPORTED_ANALYSIS_VIEWS: frozenset[str] = frozenset(
    {"all_matched", "stopwords_excluded", "content_words"}
)
SUPPORTED_WEIGHTINGS: frozenset[str] = frozenset({"token", "type"})


class VerseVADReaderError(RuntimeError):
    """Base error for reader failures."""


class VerseVADSchemaError(VerseVADReaderError):
    """Raised when an input does not satisfy the expected VerseVAD schema."""


class MetricNotFoundError(VerseVADReaderError):
    """Raised when no rows match a requested metric specification."""


class AmbiguousMetricError(VerseVADReaderError):
    """Raised when a metric specification matches more than one metric identity."""


class DuplicateMetricRowError(VerseVADReaderError):
    """Raised when one exact metric produces multiple rows for the same work."""


@dataclass(frozen=True)
class MetricSpec:
    """An exact VerseVAD poem-level metric selection.

    ``lexicon_id``, ``metric``, ``analysis_view``, and ``weighting`` are always
    required. ``dimension`` and ``category`` may be omitted only when the
    remaining fields identify exactly one metric identity. If omission is
    ambiguous, the reader refuses to guess.
    """

    lexicon_id: str
    metric: str
    analysis_view: str
    weighting: str
    dimension: Optional[str] = None
    category: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.lexicon_id.strip():
            raise ValueError("MetricSpec.lexicon_id may not be blank.")
        if not self.metric.strip():
            raise ValueError("MetricSpec.metric may not be blank.")
        if self.analysis_view not in SUPPORTED_ANALYSIS_VIEWS:
            raise ValueError(
                f"Unsupported analysis_view {self.analysis_view!r}. "
                f"Expected one of {sorted(SUPPORTED_ANALYSIS_VIEWS)}."
            )
        if self.weighting not in SUPPORTED_WEIGHTINGS:
            raise ValueError(
                f"Unsupported weighting {self.weighting!r}. "
                f"Expected one of {sorted(SUPPORTED_WEIGHTINGS)}."
            )

    @property
    def short_name(self) -> str:
        bits = [self.lexicon_id, self.metric]
        if self.dimension:
            bits.append(self.dimension)
        if self.category:
            bits.append(self.category)
        bits.extend([self.analysis_view, self.weighting])
        return " | ".join(bits)


@dataclass(frozen=True)
class ValidationReport:
    source: str
    source_kind: str
    archive_member: Optional[str]
    schema_fingerprint: str
    row_count: int
    work_count: int
    run_count: int
    lexicon_count: int
    metric_identity_count: int
    analysis_views: tuple[str, ...]
    weightings: tuple[str, ...]
    extra_columns: tuple[str, ...]
    metadata_conflicts: int
    blank_text_ids: int

    @property
    def valid(self) -> bool:
        return self.metadata_conflicts == 0 and self.blank_text_ids == 0

    def to_text(self) -> str:
        member = self.archive_member or "(standalone CSV)"
        extras = ", ".join(self.extra_columns) if self.extra_columns else "None"
        status = "VALID" if self.valid else "INVALID"
        lines = [
            "VerseVAD Corpus Reader",
            "======================",
            f"Reader version: {__version__}",
            f"Source: {self.source}",
            f"Source kind: {self.source_kind}",
            f"Metrics member: {member}",
            f"Schema: {status}",
            f"Schema fingerprint: {self.schema_fingerprint}",
            f"Rows: {self.row_count:,}",
            f"Unique works: {self.work_count:,}",
            f"Run IDs: {self.run_count:,}",
            f"Lexical resources: {self.lexicon_count:,}",
            f"Metric identities: {self.metric_identity_count:,}",
            f"Analysis views: {', '.join(self.analysis_views)}",
            f"Weightings: {', '.join(self.weightings)}",
            f"Extra columns beyond baseline: {extras}",
            f"Blank text IDs: {self.blank_text_ids:,}",
            f"Metadata conflicts by text_id: {self.metadata_conflicts:,}",
        ]
        return "\n".join(lines)


class VerseVADCorpusReader:
    """Read and standardize VerseVAD ``corpus_vad_metrics.csv`` data."""

    def __init__(self, source: str | Path, *, chunksize: int = 75_000) -> None:
        self.source = Path(source).expanduser().resolve()
        self.chunksize = int(chunksize)
        if self.chunksize <= 0:
            raise ValueError("chunksize must be a positive integer.")
        if not self.source.exists():
            raise FileNotFoundError(f"Source does not exist: {self.source}")
        if self.source.is_dir():
            raise VerseVADReaderError(
                "The reader expects a Complete Audit ZIP or a standalone CSV, not a directory."
            )
        suffix = self.source.suffix.lower()
        if suffix not in {".zip", ".csv"}:
            raise VerseVADReaderError(
                f"Unsupported source type {suffix!r}. Expected .zip or .csv."
            )

    @property
    def source_kind(self) -> str:
        return "zip" if self.source.suffix.lower() == ".zip" else "csv"

    def _find_archive_member(self, archive: zipfile.ZipFile) -> str:
        matches = [
            name
            for name in archive.namelist()
            if not name.endswith("/") and Path(name).name == METRICS_FILENAME
        ]
        if not matches:
            raise VerseVADSchemaError(
                f"ZIP does not contain {METRICS_FILENAME!r}: {self.source}"
            )
        if len(matches) > 1:
            pretty = "\n  - ".join(matches)
            raise VerseVADSchemaError(
                f"ZIP contains multiple {METRICS_FILENAME!r} files; refusing to guess:\n"
                f"  - {pretty}"
            )
        return matches[0]

    @contextmanager
    def _open_binary(self) -> Iterator[BinaryIO]:
        if self.source_kind == "csv":
            with self.source.open("rb") as handle:
                yield handle
            return

        with zipfile.ZipFile(self.source, "r") as archive:
            member = self._find_archive_member(archive)
            with archive.open(member, "r") as handle:
                yield handle

    def archive_member(self) -> Optional[str]:
        if self.source_kind == "csv":
            return None
        with zipfile.ZipFile(self.source, "r") as archive:
            return self._find_archive_member(archive)

    def header(self) -> tuple[str, ...]:
        with self._open_binary() as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            try:
                row = next(csv.reader(text))
            except StopIteration as exc:
                raise VerseVADSchemaError("Metrics CSV is empty.") from exc
            finally:
                # Detach so TextIOWrapper does not attempt to close a ZipExtFile
                # already managed by the outer context manager.
                try:
                    text.detach()
                except Exception:
                    pass
        return tuple(row)

    def schema_fingerprint(self) -> str:
        normalized = "\x1f".join(self.header()).encode("utf-8")
        return hashlib.sha256(normalized).hexdigest()[:16]

    def validate_schema(self, *, strict: bool = False) -> tuple[str, ...]:
        header = self.header()
        missing = sorted(REQUIRED_COLUMNS.difference(header))
        if missing:
            raise VerseVADSchemaError(
                "Input is missing required corpus_vad_metrics.csv columns:\n  - "
                + "\n  - ".join(missing)
            )
        if len(header) != len(set(header)):
            duplicates = sorted({col for col in header if header.count(col) > 1})
            raise VerseVADSchemaError(
                "Input contains duplicate column names:\n  - " + "\n  - ".join(duplicates)
            )
        if strict and header != BASELINE_COLUMNS:
            extra = [col for col in header if col not in BASELINE_COLUMNS]
            reordered_or_changed = header != BASELINE_COLUMNS
            raise VerseVADSchemaError(
                "Strict schema validation failed. The header differs from the pinned "
                "VerseVAD baseline.\n"
                f"Extra columns: {extra or 'None'}\n"
                f"Header order/content changed: {reordered_or_changed}"
            )
        return header

    def _iter_chunks(self, *, usecols: Optional[Sequence[str]] = None) -> Iterator[pd.DataFrame]:
        self.validate_schema(strict=False)
        with self._open_binary() as raw:
            for chunk in pd.read_csv(
                raw,
                encoding="utf-8-sig",
                dtype=str,
                keep_default_na=False,
                usecols=list(usecols) if usecols is not None else None,
                chunksize=self.chunksize,
            ):
                yield chunk

    def validate(self, *, strict_schema: bool = False) -> ValidationReport:
        """Run a streaming smoke test over the source.

        The scan checks required columns, counts works/resources/metric identities,
        and verifies that stable work metadata does not conflict for a text_id.
        It intentionally does not perform any statistical calculations.
        """

        header = self.validate_schema(strict=strict_schema)
        extra_columns = tuple(col for col in header if col not in BASELINE_COLUMNS)

        usecols = [
            "run_id",
            "text_id",
            "title",
            "author",
            "collection",
            "date_label",
            "genre",
            "lexicon_id",
            "metric",
            "dimension",
            "category",
            "analysis_view",
            "weighting",
        ]

        row_count = 0
        blank_text_ids = 0
        work_ids: set[str] = set()
        run_ids: set[str] = set()
        lexicons: set[str] = set()
        metric_identities: set[tuple[str, str, str, str]] = set()
        analysis_views: set[str] = set()
        weightings: set[str] = set()
        metadata_by_text: dict[str, tuple[str, str, str, str, str]] = {}
        conflicting_text_ids: set[str] = set()

        for chunk in self._iter_chunks(usecols=usecols):
            row_count += len(chunk)
            blank_text_ids += int((chunk["text_id"].str.strip() == "").sum())
            work_ids.update(x for x in chunk["text_id"].unique() if x)
            run_ids.update(x for x in chunk["run_id"].unique() if x)
            lexicons.update(x for x in chunk["lexicon_id"].unique() if x)
            analysis_views.update(x for x in chunk["analysis_view"].unique() if x)
            weightings.update(x for x in chunk["weighting"].unique() if x)

            identities = chunk[["lexicon_id", "metric", "dimension", "category"]].drop_duplicates()
            metric_identities.update(map(tuple, identities.itertuples(index=False, name=None)))

            metadata = chunk[
                ["text_id", "title", "author", "collection", "date_label", "genre"]
            ].drop_duplicates()
            for row in metadata.itertuples(index=False, name=None):
                text_id = row[0]
                if not text_id:
                    continue
                values = tuple(row[1:])
                previous = metadata_by_text.get(text_id)
                if previous is None:
                    metadata_by_text[text_id] = values
                elif previous != values:
                    conflicting_text_ids.add(text_id)

        unexpected_views = analysis_views.difference(SUPPORTED_ANALYSIS_VIEWS)
        unexpected_weights = weightings.difference(SUPPORTED_WEIGHTINGS)
        if unexpected_views:
            raise VerseVADSchemaError(
                f"Unexpected analysis_view value(s): {sorted(unexpected_views)}"
            )
        if unexpected_weights:
            raise VerseVADSchemaError(
                f"Unexpected weighting value(s): {sorted(unexpected_weights)}"
            )

        report = ValidationReport(
            source=str(self.source),
            source_kind=self.source_kind,
            archive_member=self.archive_member(),
            schema_fingerprint=self.schema_fingerprint(),
            row_count=row_count,
            work_count=len(work_ids),
            run_count=len(run_ids),
            lexicon_count=len(lexicons),
            metric_identity_count=len(metric_identities),
            analysis_views=tuple(sorted(analysis_views)),
            weightings=tuple(sorted(weightings)),
            extra_columns=extra_columns,
            metadata_conflicts=len(conflicting_text_ids),
            blank_text_ids=blank_text_ids,
        )

        if not report.valid:
            problems: list[str] = []
            if blank_text_ids:
                problems.append(f"{blank_text_ids:,} row(s) have blank text_id")
            if conflicting_text_ids:
                problems.append(
                    f"{len(conflicting_text_ids):,} text_id(s) have conflicting work metadata"
                )
            raise VerseVADSchemaError("Validation failed: " + "; ".join(problems))

        return report

    def catalog(self) -> pd.DataFrame:
        """Return all distinct metric/profile combinations available in the source."""

        cols = [
            "lexicon_id",
            "lexicon",
            "value_kind",
            "metric",
            "dimension",
            "category",
            "analysis_view",
            "weighting",
            "scale",
        ]
        pieces: list[pd.DataFrame] = []
        for chunk in self._iter_chunks(usecols=cols):
            pieces.append(chunk.drop_duplicates())
        if not pieces:
            return pd.DataFrame(columns=cols)
        result = pd.concat(pieces, ignore_index=True).drop_duplicates()
        return result.sort_values(
            ["lexicon_id", "metric", "dimension", "category", "analysis_view", "weighting"],
            kind="stable",
        ).reset_index(drop=True)

    def metric_definitions(self) -> pd.DataFrame:
        """Return a compact catalog without repeating scope/weighting combinations."""

        catalog = self.catalog()
        cols = ["lexicon_id", "lexicon", "value_kind", "metric", "dimension", "category", "scale"]
        if catalog.empty:
            return pd.DataFrame(columns=cols + ["profiles"])

        grouped = (
            catalog.assign(
                profile=catalog["analysis_view"] + " / " + catalog["weighting"]
            )
            .groupby(cols, dropna=False, sort=False)["profile"]
            .agg(lambda values: "; ".join(sorted(set(values))))
            .reset_index(name="profiles")
        )
        return grouped.sort_values(
            ["lexicon_id", "metric", "dimension", "category"], kind="stable"
        ).reset_index(drop=True)

    @staticmethod
    def _coerce_numeric(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
        frame = frame.copy()
        for column in columns:
            raw = frame[column].astype(str)
            converted = pd.to_numeric(raw.where(raw.str.strip() != "", pd.NA), errors="coerce")
            invalid = (raw.str.strip() != "") & converted.isna()
            if invalid.any():
                examples = raw[invalid].drop_duplicates().head(5).tolist()
                raise VerseVADSchemaError(
                    f"Column {column!r} contains nonnumeric nonblank value(s): {examples}"
                )
            frame[column] = converted
        return frame

    def select_metric(self, spec: MetricSpec) -> pd.DataFrame:
        """Return one exact metric row per work.

        The reader refuses to guess when an underspecified request matches multiple
        dimensions/categories. Missing numeric values remain missing; they are never
        replaced with zero.
        """

        usecols = list(
            dict.fromkeys(
                [
                    "run_id",
                    *METADATA_COLUMNS,
                    "lexicon_id",
                    "lexicon",
                    "value_kind",
                    "metric",
                    "dimension",
                    "category",
                    "analysis_view",
                    "weighting",
                    "scale",
                    "denominator",
                    *NUMERIC_COLUMNS,
                    "completed_at",
                ]
            )
        )

        matches: list[pd.DataFrame] = []
        for chunk in self._iter_chunks(usecols=usecols):
            mask = (
                (chunk["lexicon_id"] == spec.lexicon_id)
                & (chunk["metric"] == spec.metric)
                & (chunk["analysis_view"] == spec.analysis_view)
                & (chunk["weighting"] == spec.weighting)
            )
            if spec.dimension is not None:
                mask &= chunk["dimension"] == spec.dimension
            if spec.category is not None:
                mask &= chunk["category"] == spec.category
            selected = chunk.loc[mask]
            if not selected.empty:
                matches.append(selected)

        if not matches:
            raise MetricNotFoundError(
                "No rows matched the requested metric:\n"
                f"  {spec.short_name}\n"
                "Run the reader with --catalog to inspect available metric identities."
            )

        result = pd.concat(matches, ignore_index=True)

        identities = result[
            ["lexicon_id", "metric", "dimension", "category", "analysis_view", "weighting"]
        ].drop_duplicates()
        if len(identities) != 1:
            choices = "\n".join(
                "  - dimension={!r}, category={!r}".format(row.dimension, row.category)
                for row in identities.itertuples(index=False)
            )
            raise AmbiguousMetricError(
                "Metric specification is ambiguous. Add dimension and/or category.\n"
                f"Requested: {spec.short_name}\n"
                f"Matched identities:\n{choices}"
            )

        duplicate_mask = result.duplicated(subset=["text_id"], keep=False)
        if duplicate_mask.any():
            duplicate_ids = result.loc[duplicate_mask, "text_id"].drop_duplicates().head(10).tolist()
            raise DuplicateMetricRowError(
                "An exact metric selection produced multiple rows for the same text_id. "
                "The reader will not choose one arbitrarily. Example text_id values: "
                + ", ".join(duplicate_ids)
            )

        result = self._coerce_numeric(result, NUMERIC_COLUMNS)

        # Add semantically neutral aliases. In VerseVAD's source schema the fields
        # are named matched_tokens/lexical_tokens even on type-weighted rows.
        result["matched_observations"] = result["matched_tokens"]
        result["eligible_observations"] = result["lexical_tokens"]
        result["observation_unit"] = spec.weighting

        # Audit the denominator metadata. This catches token/type denominator drift
        # before downstream statistics ever see the data.
        eligible = result["eligible_observations"]
        matched = result["matched_observations"]
        reported = result["coverage"]
        recomputed = matched / eligible.where(eligible != 0)
        comparable = reported.notna() & recomputed.notna()
        inconsistent = comparable & ((reported - recomputed).abs() > 1e-10)
        if inconsistent.any():
            sample = result.loc[
                inconsistent,
                ["text_id", "matched_observations", "eligible_observations", "coverage"],
            ].head(5)
            raise VerseVADSchemaError(
                "Coverage metadata is inconsistent with matched/eligible observations for "
                f"{int(inconsistent.sum()):,} selected row(s). Example(s):\n"
                + sample.to_string(index=False)
            )

        result["coverage_recomputed"] = recomputed
        result["coverage_consistent"] = ~inconsistent

        preferred = [
            "text_id",
            "text_version_id",
            "title",
            "author",
            "collection",
            "date_label",
            "genre",
            "value",
            "observations",
            "matched_observations",
            "eligible_observations",
            "observation_unit",
            "coverage",
            "coverage_recomputed",
            "coverage_consistent",
            "lexicon_id",
            "lexicon",
            "metric",
            "dimension",
            "category",
            "analysis_view",
            "weighting",
            "value_kind",
            "scale",
            "denominator",
            "run_id",
            "completed_at",
            # Retain the original VerseVAD fields for auditability.
            "matched_tokens",
            "lexical_tokens",
        ]
        return result[preferred].reset_index(drop=True)

    def pair_metrics(
        self,
        x_spec: MetricSpec,
        y_spec: MetricSpec,
        *,
        drop_missing: bool = False,
    ) -> pd.DataFrame:
        """Join two exact poem-level metrics by ``text_id``.

        This is data preparation only. No correlation or other statistic is
        calculated here. By default, missing values are retained and marked with
        ``complete_pair``. Set ``drop_missing=True`` to return only complete pairs.
        """

        x = self.select_metric(x_spec)
        y = self.select_metric(y_spec)

        metadata = ["text_id", "text_version_id", "title", "author", "collection", "date_label", "genre"]
        evidence = [
            "value",
            "observations",
            "matched_observations",
            "eligible_observations",
            "observation_unit",
            "coverage",
        ]

        x_keep = x[metadata + evidence].copy()
        y_keep = y[metadata + evidence].copy()

        # Preserve the selected X metric's stable source order. Outer merges may
        # otherwise reorder keys, which is harmless statistically but annoying
        # for reproducible exported paired-data files. Y-only works, if any,
        # follow the X works in their own stable source order.
        x_keep["_x_order"] = range(len(x_keep))
        y_keep["_y_order"] = range(len(y_keep))

        x_keep = x_keep.rename(columns={col: f"x_{col}" for col in evidence})
        y_keep = y_keep.rename(columns={col: f"y_{col}" for col in evidence})

        merged = x_keep.merge(y_keep, on="text_id", how="outer", suffixes=("_xmeta", "_ymeta"), sort=False)
        merged["_pair_order"] = merged["_x_order"].where(
            merged["_x_order"].notna(), len(x_keep) + merged["_y_order"]
        )
        merged = merged.sort_values("_pair_order", kind="stable").drop(
            columns=["_x_order", "_y_order", "_pair_order"]
        )

        # Reconcile metadata and refuse silent disagreement.
        for col in metadata[1:]:
            left = f"{col}_xmeta"
            right = f"{col}_ymeta"
            left_values = merged[left].fillna("").astype(str)
            right_values = merged[right].fillna("").astype(str)
            conflict = (left_values != "") & (right_values != "") & (left_values != right_values)
            if conflict.any():
                ids = merged.loc[conflict, "text_id"].head(10).tolist()
                raise VerseVADSchemaError(
                    f"Work metadata conflict between paired metrics for field {col!r}. "
                    f"Example text_id values: {ids}"
                )
            merged[col] = left_values.where(left_values != "", right_values)
            merged = merged.drop(columns=[left, right])

        merged["complete_pair"] = merged["x_value"].notna() & merged["y_value"].notna()
        if drop_missing:
            merged = merged.loc[merged["complete_pair"]].copy()

        ordered = [
            "text_id",
            "text_version_id",
            "title",
            "author",
            "collection",
            "date_label",
            "genre",
            "x_value",
            "y_value",
            "complete_pair",
            "x_observations",
            "y_observations",
            "x_matched_observations",
            "y_matched_observations",
            "x_eligible_observations",
            "y_eligible_observations",
            "x_observation_unit",
            "y_observation_unit",
            "x_coverage",
            "y_coverage",
        ]
        return merged[ordered].reset_index(drop=True)


def _print_catalog(reader: VerseVADCorpusReader) -> None:
    definitions = reader.metric_definitions()
    if definitions.empty:
        print("No metrics found.")
        return

    current_lexicon: Optional[str] = None
    for row in definitions.itertuples(index=False):
        if row.lexicon_id != current_lexicon:
            current_lexicon = row.lexicon_id
            print()
            print(row.lexicon)
            print(f"  lexicon_id: {row.lexicon_id}")
        dim = f" | dimension={row.dimension}" if row.dimension else ""
        cat = f" | category={row.category}" if row.category else ""
        print(f"    {row.metric}{dim}{cat}")
        print(f"      profiles: {row.profiles}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and read VerseVAD corpus_vad_metrics.csv from a Complete Audit ZIP "
            "or standalone CSV."
        )
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("source", help="Path to VerseVAD Complete Audit .zip or corpus_vad_metrics.csv")
    parser.add_argument("--strict-schema", action="store_true", help="Require the exact pinned baseline header")
    parser.add_argument("--catalog", action="store_true", help="Print available metric identities and profiles")
    parser.add_argument("--catalog-csv", help="Write the full metric/profile catalog to this CSV path")

    parser.add_argument("--lexicon-id", help="Exact lexicon_id for metric extraction")
    parser.add_argument("--metric", help="Exact metric ID for metric extraction")
    parser.add_argument("--dimension", help="Optional exact dimension")
    parser.add_argument("--category", help="Optional exact category")
    parser.add_argument(
        "--analysis-view",
        choices=sorted(SUPPORTED_ANALYSIS_VIEWS),
        help="Lexical scope for metric extraction",
    )
    parser.add_argument(
        "--weighting",
        choices=sorted(SUPPORTED_WEIGHTINGS),
        help="token or type weighting for metric extraction",
    )
    parser.add_argument("--output", help="Write extracted metric rows to this CSV path")
    parser.add_argument("--chunksize", type=int, default=75_000, help="Rows per streaming chunk (default: 75000)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_console_encoding()
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        reader = VerseVADCorpusReader(args.source, chunksize=args.chunksize)
        report = reader.validate(strict_schema=args.strict_schema)
        print(report.to_text())

        if args.catalog:
            _print_catalog(reader)

        if args.catalog_csv:
            out = Path(args.catalog_csv).expanduser()
            out.parent.mkdir(parents=True, exist_ok=True)
            reader.catalog().to_csv(out, index=False)
            print(f"\nCatalog written: {out}")

        spec_fields = [args.lexicon_id, args.metric, args.analysis_view, args.weighting]
        extraction_requested = any(value is not None for value in spec_fields) or args.output is not None
        if extraction_requested:
            if not all(value is not None for value in spec_fields):
                parser.error(
                    "Metric extraction requires --lexicon-id, --metric, --analysis-view, and --weighting."
                )
            spec = MetricSpec(
                lexicon_id=args.lexicon_id,
                metric=args.metric,
                dimension=args.dimension,
                category=args.category,
                analysis_view=args.analysis_view,
                weighting=args.weighting,
            )
            selected = reader.select_metric(spec)
            valid_values = int(selected["value"].notna().sum())
            print(
                f"\nSelected metric: {spec.short_name}\n"
                f"Rows/works: {len(selected):,}\n"
                f"Nonmissing values: {valid_values:,}"
            )
            preview_cols = ["text_id", "title", "value", "observations", "coverage"]
            print("\nPreview:")
            print(selected[preview_cols].head(10).to_string(index=False))
            if args.output:
                out = Path(args.output).expanduser()
                out.parent.mkdir(parents=True, exist_ok=True)
                selected.to_csv(out, index=False)
                print(f"\nMetric export written: {out}")

        return 0
    except (VerseVADReaderError, FileNotFoundError, ValueError, zipfile.BadZipFile) as exc:
        print(f"VERSEVAD READER ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
