#!/usr/bin/env python3
"""anomaly.py

Interactive exploratory anomaly finder for VerseVAD corpus exports.

Designed to live beside ``versevad_reader.py`` inside::

    versevad_stats/
        scripts/
            versevad_reader.py
            correlation.py
            anomaly.py
        source/
            <VerseVAD Corpus / Research Project Complete Audit ZIPs>
        exports/

Run from the ``versevad_stats`` project folder with::

    python scripts/anomaly.py

Data tables
-----------
When a Complete Audit ZIP contains both supported corpus tables, the script
asks which evidence layer to use:

1. ``corpus_vad_metrics.csv``
   The profile-aware lexical/normative table used by ``correlation.py``.
   This includes VAD, concreteness, AoA, frequency, sensorimotor, emotion,
   lexical-style metrics, coverage, token/type weighting, and the three lexical
   scopes exposed by VerseVAD.

2. ``corpus_module_metrics.csv``
   The broader poem-level module table. This makes exploratory anomaly searches
   available for numeric document-level measures from rhyme/phonology, meter,
   prosody, readability, lexical diversity/style, inherited-form analysis,
   VADER, PoetryID, lexical SD/IQR measures, and other module summaries.
   A small set of technical/audit-only fields (for example raw VerseMap PCA
   coordinates and stable IDs) is intentionally excluded from the anomaly menu.

The full module table contains heterogeneous metrics, so there is no single
honest poem-level "coverage" field that can be applied across rhyme, meter,
readability, lexical norms, and form. Therefore the familiar coverage threshold
is offered only for ``corpus_vad_metrics.csv``. Module-specific denominators,
notes, units, and evidence fields are retained in the exports.

Analysis modes
--------------
1. Single-metric extremes
   Highest, lowest, or both tails for one selected numeric poem-level metric.

2. Two-metric directional combinations
   HIGH/HIGH, HIGH/LOW, LOW/HIGH, or LOW/LOW combinations. Directional
   extremity is defined with corpus-relative percentile ranks, and BOTH metrics
   must satisfy the user-selected percentile threshold.

3. Broad anomaly scan
   A descriptive discovery pass that identifies poems with unusually extreme
   multi-metric profiles and emits short explanations of what is unusual.

   * Lexical/VAD mode uses semantically useful mean/rate/dispersion metrics,
     collapses parallel resources for the same concept by median percentile,
     and excludes cumulative/load and coverage fields from the automatic score.
   * Full-module mode uses numeric document-level summaries including SD/IQR and
     rhyme/meter/readability/form measures, balances the broad profile by module,
     excludes technical/audit-only fields and cumulative/load metrics from the
     automatic score, and can also surface rare categorical form/meter labels
     when the category space is small enough to be interpretable.

This tool is deliberately EXPLORATORY. "Anomalous" means unusual relative to
other works in the selected corpus/profile. It is not a significance test and
does not imply error, pathology, artistic value, or causal importance.

Exports
-------
Every run writes to ``exports/anomalies/<run>/`` and includes:

* anomaly_results.csv
* anomaly_analysis.xlsx
* analysis_spec.json
* analysis_metadata.json
* a mode-specific full ranking/evidence CSV

Dependencies
------------
Python 3.10+, numpy, pandas, scipy, and versevad_reader.py.
The XLSX writer uses only Python's standard library.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
import zipfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Iterator, Optional, Sequence
from xml.sax.saxutils import escape as xml_escape

try:
    import numpy as np
    import pandas as pd
    from scipy.stats import rankdata
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "anomaly.py requires numpy, pandas, and scipy.\n"
        "Install them with:\n"
        "  python -m pip install numpy pandas scipy"
    ) from exc

try:
    import versevad_reader
    from versevad_reader import MetricSpec, VerseVADCorpusReader, VerseVADReaderError
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "anomaly.py could not import versevad_reader.py. Put both files "
        "in the same scripts folder."
    ) from exc

from versevad_tools.core import configure_console_encoding, project_root as shared_project_root
from versevad_tools.audit import AuditSourceError, require_audit
from versevad_tools.sources import choose_one_source, discover_files


__version__ = "0.2.0"
VAD_METRICS_FILENAME = "corpus_vad_metrics.csv"
MODULE_METRICS_FILENAME = "corpus_module_metrics.csv"
MAX_SEARCH_RESULTS = 20
DEFAULT_RESULT_COUNT = 20
DEFAULT_PAIR_EXTREMITY = 75.0
GENERAL_TOP_COMPONENTS = 3
GENERAL_MIN_CONCEPTS = 8
GENERAL_MIN_METRIC_N = 20
MODULE_GENERAL_MIN_MODULES = 5
MODULE_RARE_CATEGORY_MAX_LEVELS = 20
MODULE_RARE_CATEGORY_MAX_SHARE = 0.10
MODULE_CHUNKSIZE = 75_000

VIEW_LABELS = {
    "content_words": "Content words only",
    "stopwords_excluded": "Stopword-excluded",
    "all_matched": "All lexical tokens",
}
WEIGHT_LABELS = {"token": "Token-weighted", "type": "Type-weighted"}

MODULE_REQUIRED_COLUMNS = (
    "run_id",
    "text_id",
    "text_version_id",
    "title",
    "author",
    "collection",
    "date_label",
    "genre",
    "module_name",
    "module_version",
    "result_id",
    "configuration_id",
    "metric_id",
    "value",
    "layer",
    "scope",
    "scope_id",
    "unit",
    "weighting",
    "denominator",
    "observation_count",
    "note",
    "completed_at",
)

MODULE_PROFILE_OPTIONS = (
    ("all_matched", "token", "All lexical tokens · Token-weighted"),
    ("all_matched", "type", "All lexical tokens · Type-weighted"),
    ("stopwords_excluded", "token", "Stopword-excluded · Token-weighted"),
    ("stopwords_excluded", "type", "Stopword-excluded · Type-weighted"),
    ("content_words", "token", "Content words only · Token-weighted"),
    ("content_words", "type", "Content words only · Type-weighted"),
)

MODULE_NAMES = {
    "age_of_acquisition": "Age of Acquisition",
    "candidate_meter_and_rhythmic_regularity": "Meter / Rhythmic Regularity",
    "concreteness": "Concreteness",
    "inherited_form": "Inherited Form",
    "lexical_frequency": "Lexical Frequency",
    "lexical_style": "Lexical Style / Diversity",
    "poetry_id": "PoetryID",
    "pronunciation_prosody_foundation": "Pronunciation / Prosody",
    "readability": "Readability",
    "rhyme_and_phonological_patterns": "Rhyme / Phonology",
    "sensorimotor_imagery_and_embodiment": "Sensorimotor / Embodiment",
    "vader_sentiment": "VADER Sentiment",
    "versemap": "VerseMap",
}

MODULE_METRIC_LABELS = {
    "phonology.alliteration_density": "Alliteration density",
    "phonology.assonance_density": "Assonance density",
    "phonology.consonance_density": "Consonance density",
    "phonology.internal_rhyme_pair_count": "Internal-rhyme pair count",
    "phonology.rhyme_density": "Rhyme density",
    "phonology.slant_rhyme_pair_count": "Slant-rhyme pair count",
    "phonology.rhyme_scheme": "Rhyme scheme",
    "meter.matching_line_proportion": "Matching-line proportion",
    "meter.performance.mean_realized_score": "Mean realized rhythmic score",
    "meter.rhythmic_variability": "Rhythmic variability (SD)",
    "meter.whole_poem_mean_fit": "Whole-poem meter fit",
    "meter.closest_candidate": "Closest meter candidate",
    "meter.performance.rhythmic_organization": "Rhythmic organization",
    "pronunciation.mean_syllables_per_complete_line": "Mean syllables per complete line",
    "pronunciation.mean_syllables_per_resolved_word": "Mean syllables per resolved word",
    "pronunciation.median_syllables_per_complete_line": "Median syllables per complete line",
    "pronunciation.median_syllables_per_resolved_word": "Median syllables per resolved word",
    "pronunciation.stress_density": "Lexical stress density",
    "readability.flesch_reading_ease": "Flesch Reading Ease",
    "readability.flesch_kincaid_grade": "Flesch-Kincaid Grade",
    "readability.gunning_fog_index": "Gunning Fog Index",
    "readability.automated_readability_index": "Automated Readability Index",
    "readability.coleman_liau_index": "Coleman-Liau Index",
    "readability.smog_index": "SMOG Index",
    "readability.poetic_reading_ease.score": "VV-PRE score",
    "lexical_style.hdd": "HDD lexical diversity",
    "lexical_style.mattr": "MATTR lexical diversity",
    "lexical_style.mtld": "MTLD lexical diversity",
    "lexical_style.mean_word_length": "Mean word length",
    "lexical_style.mean_words_per_nonblank_line": "Mean words per nonblank line",
    "lexical_style.mean_words_per_stanza": "Mean words per stanza",
    "lexical_style.mean_nonblank_lines_per_stanza": "Mean nonblank lines per stanza",
    "lexical_style.population_sd_words_per_nonblank_line": "Words-per-line variability (SD)",
    "lexical_style.population_sd_words_per_stanza": "Words-per-stanza variability (SD)",
    "lexical_style.population_sd_nonblank_lines_per_stanza": "Lines-per-stanza variability (SD)",
    "vader.document.compound_score": "VADER compound sentiment",
    "vader.document.positive_proportion": "VADER positive proportion",
    "vader.document.neutral_proportion": "VADER neutral proportion",
    "vader.document.negative_proportion": "VADER negative proportion",
    "inherited_form.best_consistency": "Best inherited-form consistency",
    "inherited_form.candidate_margin": "Inherited-form candidate margin",
    "poetry_id.centroid_distance": "PoetryID centroid distance",
    "poetry_id.neighbor_margin": "PoetryID neighbor margin",
    "poetry_id.valence": "PoetryID valence",
    "poetry_id.arousal": "PoetryID arousal",
    "poetry_id.dominance": "PoetryID dominance",
}

# Technical/audit-only values should not become reader-facing anomaly metrics.
MODULE_HARD_EXCLUDE_PATTERNS = (
    "versemap.coordinate_",
    "source_vad_result_id",
    "result_id",
    "configuration_id",
)

# General scan exclusions. These remain searchable/selectable in single/pair mode
# unless hard-excluded, but are not automatically used for the broad profile.
MODULE_GENERAL_EXCLUDE_PATTERNS = (
    "coverage",
    "evidence_weight",
    "minimum_component_coverage",
    "minimum_lexical_matched_count",
    "cumulative_load",
    "load_per_100",
    "coordinate_",
)

MODULE_SIZE_PATTERNS = (
    "word_count",
    "token_count",
    "sentence_count",
    "syllable_count",
    "surface_type_count",
    "internal_rhyme_pair_count",
    "slant_rhyme_pair_count",
)


@dataclass(frozen=True)
class LexicalMetricChoice:
    label: str
    lexicon_id: str
    lexicon: str
    metric: str
    dimension: str
    category: str
    scale: str
    analysis_view: str
    weighting: str

    def to_spec(self) -> MetricSpec:
        return MetricSpec(
            lexicon_id=self.lexicon_id,
            metric=self.metric,
            dimension=self.dimension or None,
            category=self.category or None,
            analysis_view=self.analysis_view,
            weighting=self.weighting,
        )

    @property
    def identity_key(self) -> tuple[str, ...]:
        return (
            self.lexicon_id,
            self.metric,
            self.dimension,
            self.category,
            self.analysis_view,
            self.weighting,
        )

    @property
    def concept_label(self) -> str:
        return self.label.split("·", 1)[0].strip()

    @property
    def resource_label(self) -> str:
        return self.label.split("·", 1)[1].strip() if "·" in self.label else self.lexicon


@dataclass(frozen=True)
class ModuleMetricChoice:
    label: str
    module_name: str
    module_version: str
    metric_id: str
    unit: str
    weighting: str
    scope_id: str
    layer: str
    note: str
    numeric: bool = True

    @property
    def identity_key(self) -> tuple[str, ...]:
        return (
            self.module_name,
            self.metric_id,
            self.unit,
            self.weighting,
            self.scope_id,
            self.layer,
        )

    @property
    def concept_label(self) -> str:
        return self.label.split("·", 1)[0].strip()

    @property
    def resource_label(self) -> str:
        return MODULE_NAMES.get(self.module_name, _pretty_words(self.module_name))

    @property
    def concept_key(self) -> str:
        return f"{self.module_name}|{self.metric_id}"


@dataclass(frozen=True)
class ModuleValidationReport:
    source: str
    source_kind: str
    archive_member: Optional[str]
    schema_fingerprint: str
    row_count: int
    work_count: int
    module_count: int
    metric_id_count: int


# ---------------------------------------------------------------------------
# Paths and source/table discovery
# ---------------------------------------------------------------------------


def project_root() -> Path:
    return shared_project_root(__file__)


def source_directory(root: Path) -> Path:
    return root / "source"


def export_directory(root: Path) -> Path:
    return root / "exports" / "anomalies"


def discover_sources(directory: Path) -> list[Path]:
    return discover_files(directory)


def choose_source_interactively(sources: Sequence[Path]) -> Path:
    return choose_one_source(sources)


def _zip_member_by_basename(source: Path, basename: str) -> Optional[str]:
    with zipfile.ZipFile(source) as archive:
        matches = [n for n in archive.namelist() if Path(n).name.casefold() == basename.casefold()]
    if len(matches) > 1:
        raise ValueError(f"Archive contains multiple files named {basename}: {matches}")
    return matches[0] if matches else None


def _csv_header(source: Path) -> list[str]:
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader(handle))


def available_tables(source: Path) -> dict[str, str]:
    """Return supported table keys and their member/path labels."""

    if source.suffix.lower() == ".zip":
        try:
            require_audit(
                source,
                expected_analysis_mode="corpus",
                require_complete=True,
            )
        except AuditSourceError as exc:
            raise ValueError(str(exc)) from exc
        tables: dict[str, str] = {}
        vad = _zip_member_by_basename(source, VAD_METRICS_FILENAME)
        module = _zip_member_by_basename(source, MODULE_METRICS_FILENAME)
        if vad:
            tables["vad"] = vad
        if module:
            tables["module"] = module
        return tables

    # Retained only as a legacy programmatic adapter. Interactive discovery
    # offers Complete Audit ZIPs exclusively.
    if source.suffix.lower() == ".csv":
        header = set(_csv_header(source))
        if set(getattr(versevad_reader, "REQUIRED_COLUMNS", ())).issubset(header):
            return {"vad": source.name}
        if set(MODULE_REQUIRED_COLUMNS).issubset(header):
            return {"module": source.name}
        return {}

    return {}


def choose_table_interactively(source: Path, tables: dict[str, str], forced: Optional[str]) -> str:
    if not tables:
        raise ValueError(
            f"{source.name} does not contain either supported table: "
            f"{VAD_METRICS_FILENAME} or {MODULE_METRICS_FILENAME}."
        )
    if forced:
        if forced not in tables:
            raise ValueError(f"Requested --table {forced!r}, but that table is not available in {source.name}.")
        return forced
    if len(tables) == 1:
        key = next(iter(tables))
        label = VAD_METRICS_FILENAME if key == "vad" else MODULE_METRICS_FILENAME
        print(f"Detected analysis table: {label}\n")
        return key

    print("Choose anomaly evidence table:")
    print(f"[1] Lexical / normative metrics · {VAD_METRICS_FILENAME} [default]")
    print("    VAD, concreteness, AoA, frequency, sensorimotor, emotions, lexical style;")
    print("    profile-aware coverage + token/type weighting.")
    print(f"[2] Full module metrics · {MODULE_METRICS_FILENAME}")
    print("    Adds rhyme, meter, prosody, readability, lexical diversity, form, SD/IQR,")
    print("    VADER, PoetryID, and other poem-level module summaries.")
    while True:
        raw = input("> ").strip()
        if raw in {"", "1"}:
            return "vad"
        if raw == "2":
            return "module"
        print("Choose 1 or 2.")


# ---------------------------------------------------------------------------
# General prompts
# ---------------------------------------------------------------------------


def prompt_choice(title: str, options: Sequence[tuple[str, str]], *, default: int = 1) -> str:
    if title:
        print(title)
    for idx, (_, label) in enumerate(options, start=1):
        suffix = " [default]" if idx == default else ""
        print(f"[{idx}] {label}{suffix}")
    while True:
        raw = input("> ").strip()
        if raw == "":
            return options[default - 1][0]
        try:
            number = int(raw)
        except ValueError:
            print("Enter one of the numbered choices.")
            continue
        if 1 <= number <= len(options):
            return options[number - 1][0]
        print(f"Choose a number from 1 to {len(options)}.")


def prompt_yes_no(prompt: str, *, default: bool = False) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{hint}]: ").strip().casefold()
        if raw == "":
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Enter Y or N.")


def prompt_result_count(default: int = DEFAULT_RESULT_COUNT) -> int:
    while True:
        raw = input(f"Number of poems to return [default {default}]: ").strip()
        if raw == "":
            return default
        try:
            value = int(raw)
        except ValueError:
            print("Enter a whole number.")
            continue
        if value < 1:
            print("Enter a number of at least 1.")
            continue
        return value


def prompt_coverage_threshold() -> Optional[float]:
    print("\nMinimum poem-level coverage threshold")
    print("A poem is eligible for a lexical metric only when that metric meets this threshold.")
    print("Enter 80 for 80%, or 0.80 as a proportion. Press Enter for no extra filter.")
    while True:
        raw = input("Coverage threshold: ").strip()
        if raw == "":
            return None
        try:
            value = float(raw)
        except ValueError:
            print("Enter a number such as 80 or 0.80, or press Enter for none.")
            continue
        if value > 1:
            value /= 100.0
        if not 0 <= value <= 1:
            print("Coverage must be between 0 and 100%.")
            continue
        return value


def prompt_pair_extremity(default: float = DEFAULT_PAIR_EXTREMITY) -> float:
    print("\nDirectional percentile threshold")
    print(
        "A poem must meet the requested direction on BOTH metrics. For example, "
        "75 means high = at/above the 75th percentile and low = at/below the "
        "25th percentile."
    )
    while True:
        raw = input(f"Minimum directional extremity [default {default:.0f}]: ").strip()
        if raw == "":
            return default
        try:
            value = float(raw)
        except ValueError:
            print("Enter a number from 50 through 99.9.")
            continue
        if 50 <= value < 100:
            return value
        print("Enter a number from 50 through 99.9.")


def choose_lexical_profile() -> tuple[str, str]:
    print("Choose lexical scope:")
    view = prompt_choice(
        "",
        [
            ("content_words", "Content words only"),
            ("stopwords_excluded", "Stopword-excluded"),
            ("all_matched", "All lexical tokens"),
        ],
        default=1,
    )
    print("\nChoose weighting:")
    weighting = prompt_choice(
        "",
        [("token", "Token-weighted"), ("type", "Type-weighted")],
        default=1,
    )
    return view, weighting


def choose_module_profile() -> tuple[str, str]:
    print("\nPreferred profile for module metrics that provide profile-specific variants:")
    print("Profile-independent metrics such as rhyme density, meter fit, and readability")
    print("remain available regardless of this choice.")
    options = [(f"{scope}|{weight}", label) for scope, weight, label in MODULE_PROFILE_OPTIONS]
    selected = prompt_choice("", options, default=3)
    return tuple(selected.split("|", 1))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Lexical metric catalog/search
# ---------------------------------------------------------------------------


def _pretty_words(value: str) -> str:
    value = value.replace("_", " ").replace(".", " ").strip()
    return " ".join(part.capitalize() for part in value.split())


def _resource_short_name(lexicon_id: str, lexicon: str) -> str:
    known = {
        "brysbaert-concreteness-2014": "Brysbaert Concreteness",
        "kuperman-aoa-2012-erratum-supplement": "Kuperman AoA",
        "subtlex-us-zipf-official": "SUBTLEX-US Zipf",
        "lancaster-sensorimotor-2020": "Lancaster Sensorimotor",
        "nrc_vad_v1": "NRC VAD v1",
        "nrc_vad_v2_1": "NRC VAD v2.1",
        "warriner_vad_2013": "Warriner VAD",
        "nrc_emotion_v0_92": "NRC Emotion Association",
        "nrc_emotion_intensity_v1": "NRC Emotion Intensity",
        "versevad_lexical_style": "VerseVAD Lexical Style",
    }
    return known.get(lexicon_id, lexicon)


def _statistic_suffix(metric: str) -> str:
    replacements = {
        "vad_average_deviation_from_poem_mean": "mean absolute deviation",
        "vad_above_midpoint_load": "above-midpoint load",
        "vad_below_midpoint_load": "below-midpoint load",
        "vad_net_midpoint_load": "net midpoint load",
        "vad_absolute_midpoint_load": "absolute midpoint load",
        "vad_above_midpoint_load_per_observation": "above-midpoint load per observation",
        "vad_below_midpoint_load_per_observation": "below-midpoint load per observation",
        "vad_net_midpoint_load_per_observation": "net midpoint load per observation",
        "vad_absolute_midpoint_load_per_observation": "absolute midpoint load per observation",
        "vad_above_midpoint_load_per_100_observations": "above-midpoint load per 100 observations",
        "vad_below_midpoint_load_per_100_observations": "below-midpoint load per 100 observations",
        "vad_net_midpoint_load_per_100_observations": "net midpoint load per 100 observations",
        "vad_absolute_midpoint_load_per_100_observations": "absolute midpoint load per 100 observations",
    }
    if metric in replacements:
        return replacements[metric]
    if metric.endswith("_standard_deviation") or metric == "vad_standard_deviation":
        return "SD"
    if metric.endswith("_mean") or metric == "vad_mean" or metric.endswith("_mean_mean"):
        return "mean"
    if metric.endswith("_cumulative"):
        return "cumulative"
    return _pretty_words(metric)


def friendly_lexical_metric_label(row: pd.Series) -> str:
    lexicon_id = str(row["lexicon_id"])
    lexicon = str(row["lexicon"])
    metric = str(row["metric"])
    dimension = str(row["dimension"] or "")
    category = str(row["category"] or "")
    resource = _resource_short_name(lexicon_id, lexicon)
    suffix = _statistic_suffix(metric)

    if lexicon_id == "brysbaert-concreteness-2014":
        base = "Concreteness"
    elif lexicon_id == "kuperman-aoa-2012-erratum-supplement":
        base = "Age of Acquisition (AoA)"
    elif lexicon_id == "subtlex-us-zipf-official":
        base = "Frequency (Zipf)"
    elif lexicon_id == "versevad_lexical_style" and dimension == "mean_word_length":
        base = "Mean word length"
    elif lexicon_id == "lancaster-sensorimotor-2020":
        base = f"{_pretty_words(dimension)} sensorimotor"
    elif lexicon_id in {"nrc_vad_v1", "nrc_vad_v2_1", "warriner_vad_2013"}:
        base = _pretty_words(dimension)
    elif lexicon_id == "nrc_emotion_v0_92":
        base = f"{_pretty_words(dimension.replace('_association', ''))} association"
    elif lexicon_id == "nrc_emotion_intensity_v1":
        base = f"{_pretty_words(dimension.replace('_intensity', ''))} intensity"
    else:
        base = _pretty_words(dimension or category or metric)

    return f"{base} {suffix}".strip() + f" · {resource}"


def is_lexical_general_metric(metric: str) -> bool:
    """Semantic eligibility for the broad lexical anomaly scan.

    Means/rates and dispersion are useful. Cumulative/load and coverage fields
    are excluded because they are often driven by text length or evidence volume.
    """

    if metric in {"coverage", "type_coverage"}:
        return False
    excluded = (
        "cumulative",
        "midpoint_load",
        "per_100_observations",
        "per_observation",
    )
    if any(token in metric for token in excluded):
        return False
    return (
        metric == "vad_mean"
        or metric == "vad_standard_deviation"
        or metric == "vad_average_deviation_from_poem_mean"
        or metric.endswith("_mean_mean")
        or metric.endswith("_mean")
        or metric.endswith("_standard_deviation")
    )


def _lexical_metric_priority(choice: LexicalMetricChoice) -> tuple[int, str]:
    if choice.metric.endswith("_mean") or choice.metric == "vad_mean" or choice.metric.endswith("_mean_mean"):
        priority = 0
    elif "standard_deviation" in choice.metric or "deviation" in choice.metric:
        priority = 1
    elif choice.metric.endswith("_cumulative") or "load" in choice.metric:
        priority = 2
    else:
        priority = 3
    return priority, choice.label.casefold()


def available_lexical_choices(
    reader: VerseVADCorpusReader, analysis_view: str, weighting: str
) -> list[LexicalMetricChoice]:
    catalog = reader.catalog()
    selected = catalog.loc[
        (catalog["analysis_view"] == analysis_view)
        & (catalog["weighting"] == weighting)
        & (~catalog["metric"].isin(["coverage", "type_coverage"]))
    ].copy()
    cols = ["lexicon_id", "lexicon", "metric", "dimension", "category", "scale"]
    selected = selected[cols].drop_duplicates().fillna("")
    choices: list[LexicalMetricChoice] = []
    for _, row in selected.iterrows():
        choices.append(
            LexicalMetricChoice(
                label=friendly_lexical_metric_label(row),
                lexicon_id=str(row["lexicon_id"]),
                lexicon=str(row["lexicon"]),
                metric=str(row["metric"]),
                dimension=str(row["dimension"]),
                category=str(row["category"]),
                scale=str(row["scale"]),
                analysis_view=analysis_view,
                weighting=weighting,
            )
        )
    return sorted(choices, key=_lexical_metric_priority)


# ---------------------------------------------------------------------------
# Full module metrics reader/catalog
# ---------------------------------------------------------------------------


class ModuleMetricsReader:
    """Strict streaming reader for poem-level ``corpus_module_metrics.csv``."""

    def __init__(self, source: str | Path, *, chunksize: int = MODULE_CHUNKSIZE):
        self.source = Path(source).expanduser().resolve()
        self.chunksize = chunksize
        if not self.source.exists():
            raise FileNotFoundError(self.source)
        if self.source.suffix.lower() not in {".zip", ".csv"}:
            raise ValueError(
                "ModuleMetricsReader requires a Complete Audit ZIP."
            )
        self.source_kind = "zip" if self.source.suffix.lower() == ".zip" else "csv"
        if self.source_kind == "zip":
            try:
                require_audit(
                    self.source,
                    expected_analysis_mode="corpus",
                    require_complete=True,
                )
            except AuditSourceError as exc:
                raise ValueError(str(exc)) from exc
        self.archive_member = (
            _zip_member_by_basename(self.source, MODULE_METRICS_FILENAME)
            if self.source_kind == "zip"
            else None
        )
        if self.source_kind == "zip" and not self.archive_member:
            raise ValueError(f"Archive does not contain {MODULE_METRICS_FILENAME}.")
        self._catalog_cache: Optional[pd.DataFrame] = None
        self._category_counts: dict[tuple[str, ...], Counter[str]] = {}

    @contextmanager
    def _open_binary(self) -> Iterator[BinaryIO]:
        if self.source_kind == "csv":
            with self.source.open("rb") as handle:
                yield handle
            return
        with zipfile.ZipFile(self.source) as archive:
            assert self.archive_member is not None
            with archive.open(self.archive_member, "r") as handle:
                yield handle

    def _header(self) -> list[str]:
        with self._open_binary() as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            return next(csv.reader(text))

    def _iter_chunks(self, *, usecols: Optional[Sequence[str]] = None) -> Iterator[pd.DataFrame]:
        with self._open_binary() as raw:
            for chunk in pd.read_csv(
                raw,
                dtype=str,
                keep_default_na=False,
                na_filter=False,
                chunksize=self.chunksize,
                usecols=usecols,
            ):
                yield chunk

    def validate(self) -> ModuleValidationReport:
        header = self._header()
        missing = [c for c in MODULE_REQUIRED_COLUMNS if c not in header]
        if missing:
            raise ValueError(f"{MODULE_METRICS_FILENAME} is missing required columns: {missing}")
        if len(header) != len(set(header)):
            raise ValueError("Module metrics CSV contains duplicate column names.")
        fingerprint = hashlib.sha256("\x1f".join(header).encode("utf-8")).hexdigest()[:16]

        row_count = 0
        works: set[str] = set()
        modules: set[str] = set()
        metrics: set[str] = set()
        for chunk in self._iter_chunks(usecols=["text_id", "module_name", "metric_id"]):
            row_count += len(chunk)
            works.update(v for v in chunk["text_id"] if v)
            modules.update(v for v in chunk["module_name"] if v)
            metrics.update(v for v in chunk["metric_id"] if v)
        return ModuleValidationReport(
            source=str(self.source),
            source_kind=self.source_kind,
            archive_member=self.archive_member,
            schema_fingerprint=fingerprint,
            row_count=row_count,
            work_count=len(works),
            module_count=len(modules),
            metric_id_count=len(metrics),
        )

    @staticmethod
    def _identity_from_row(row: pd.Series) -> tuple[str, ...]:
        return (
            str(row["module_name"]),
            str(row["module_version"]),
            str(row["metric_id"]),
            str(row["unit"]),
            str(row["weighting"]),
            str(row["scope_id"]),
            str(row["layer"]),
        )

    def catalog(self) -> pd.DataFrame:
        if self._catalog_cache is not None:
            return self._catalog_cache.copy()

        usecols = [
            "text_id",
            "module_name",
            "module_version",
            "metric_id",
            "value",
            "layer",
            "scope",
            "scope_id",
            "unit",
            "weighting",
            "note",
        ]
        stats: dict[tuple[str, ...], dict[str, object]] = {}
        category_counts: dict[tuple[str, ...], Counter[str]] = {}
        for chunk in self._iter_chunks(usecols=usecols):
            chunk = chunk.loc[chunk["scope"].eq("document")].copy()
            if chunk.empty:
                continue
            for _, row in chunk.iterrows():
                identity = self._identity_from_row(row)
                state = stats.setdefault(
                    identity,
                    {
                        "module_name": row["module_name"],
                        "module_version": row["module_version"],
                        "metric_id": row["metric_id"],
                        "unit": row["unit"],
                        "weighting": row["weighting"],
                        "scope_id": row["scope_id"],
                        "layer": row["layer"],
                        "note": row["note"],
                        "rows": 0,
                        "works": set(),
                        "duplicate_work": False,
                        "numeric_nonblank": 0,
                        "nonblank": 0,
                    },
                )
                state["rows"] = int(state["rows"]) + 1
                text_id = str(row["text_id"])
                works = state["works"]
                assert isinstance(works, set)
                if text_id in works:
                    state["duplicate_work"] = True
                works.add(text_id)
                raw = str(row["value"]).strip()
                if raw:
                    state["nonblank"] = int(state["nonblank"]) + 1
                    try:
                        value = float(raw)
                        if math.isfinite(value):
                            state["numeric_nonblank"] = int(state["numeric_nonblank"]) + 1
                    except ValueError:
                        pass
                    category_counts.setdefault(identity, Counter())[raw] += 1

        records: list[dict[str, object]] = []
        for identity, state in stats.items():
            nonblank = int(state["nonblank"])
            numeric_nonblank = int(state["numeric_nonblank"])
            numeric = nonblank > 0 and numeric_nonblank == nonblank
            counts = category_counts.get(identity, Counter())
            records.append(
                {
                    **{k: v for k, v in state.items() if k != "works"},
                    "works": len(state["works"]),
                    "numeric": numeric,
                    "unique_values": len(counts),
                }
            )
        self._category_counts = category_counts
        self._catalog_cache = pd.DataFrame(records).fillna("")
        return self._catalog_cache.copy()

    def category_counts(self, choice: ModuleMetricChoice) -> Counter[str]:
        if self._catalog_cache is None:
            self.catalog()
        key = (
            choice.module_name,
            choice.module_version,
            choice.metric_id,
            choice.unit,
            choice.weighting,
            choice.scope_id,
            choice.layer,
        )
        return self._category_counts.get(key, Counter()).copy()

    def select_metric(self, choice: ModuleMetricChoice) -> pd.DataFrame:
        if not choice.numeric:
            raise ValueError("select_metric is for numeric module metrics.")
        usecols = [
            "text_id",
            "text_version_id",
            "title",
            "author",
            "collection",
            "date_label",
            "genre",
            "module_name",
            "module_version",
            "metric_id",
            "value",
            "layer",
            "scope",
            "scope_id",
            "unit",
            "weighting",
            "denominator",
            "observation_count",
            "note",
        ]
        parts: list[pd.DataFrame] = []
        for chunk in self._iter_chunks(usecols=usecols):
            mask = (
                chunk["scope"].eq("document")
                & chunk["module_name"].eq(choice.module_name)
                & chunk["module_version"].eq(choice.module_version)
                & chunk["metric_id"].eq(choice.metric_id)
                & chunk["unit"].eq(choice.unit)
                & chunk["weighting"].eq(choice.weighting)
                & chunk["scope_id"].eq(choice.scope_id)
                & chunk["layer"].eq(choice.layer)
            )
            part = chunk.loc[mask].copy()
            if not part.empty:
                parts.append(part)
        if not parts:
            raise VerseVADReaderError(f"No rows found for module metric {choice.label}.")
        out = pd.concat(parts, ignore_index=True)
        raw = out["value"].astype(str).str.strip()
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        invalid = raw.ne("") & out["value"].isna()
        if invalid.any():
            raise VerseVADReaderError(f"Non-numeric values found in numeric module metric {choice.label}.")
        dup = out.duplicated("text_id", keep=False)
        if dup.any():
            sample = out.loc[dup, ["text_id", "title", "metric_id", "scope_id", "weighting"]].head(10)
            raise VerseVADReaderError(
                "Exact module metric produced multiple rows for the same poem:\n" + sample.to_string(index=False)
            )
        out["coverage"] = np.nan
        out["observations"] = pd.to_numeric(out["observation_count"], errors="coerce")
        out["matched_observations"] = np.nan
        out["eligible_observations"] = np.nan
        out["observation_unit"] = out["weighting"].where(out["weighting"].ne(""), np.nan)
        return out

    def pair_metrics(self, a: ModuleMetricChoice, b: ModuleMetricChoice) -> pd.DataFrame:
        left = self.select_metric(a).copy()
        right = self.select_metric(b).copy()
        meta = ["text_id", "text_version_id", "title", "author", "collection", "date_label", "genre"]
        left_keep = meta + ["value", "denominator", "observation_count", "note", "unit", "weighting", "scope_id"]
        right_keep = meta + ["value", "denominator", "observation_count", "note", "unit", "weighting", "scope_id"]
        merged = left[left_keep].merge(
            right[right_keep], on="text_id", how="outer", suffixes=("_x", "_y"), sort=False
        )
        for col in meta[1:]:
            xcol, ycol = f"{col}_x", f"{col}_y"
            if xcol in merged and ycol in merged:
                mismatch = merged[xcol].notna() & merged[ycol].notna() & (merged[xcol] != merged[ycol])
                if mismatch.any():
                    raise VerseVADReaderError(f"Metadata mismatch while pairing module metrics: {col}")
                merged[col] = merged[xcol].where(merged[xcol].notna(), merged[ycol])
                merged.drop(columns=[xcol, ycol], inplace=True)
        merged.rename(
            columns={
                "value_x": "x_value",
                "value_y": "y_value",
                "denominator_x": "x_denominator",
                "denominator_y": "y_denominator",
                "observation_count_x": "x_observation_count",
                "observation_count_y": "y_observation_count",
                "note_x": "x_note",
                "note_y": "y_note",
                "unit_x": "x_unit",
                "unit_y": "y_unit",
                "weighting_x": "x_weighting",
                "weighting_y": "y_weighting",
                "scope_id_x": "x_scope_id",
                "scope_id_y": "y_scope_id",
            },
            inplace=True,
        )
        merged["x_coverage"] = np.nan
        merged["y_coverage"] = np.nan
        merged["complete_pair"] = merged["x_value"].notna() & merged["y_value"].notna()
        return merged

    def batch_select_numeric(self, choices: Sequence[ModuleMetricChoice]) -> pd.DataFrame:
        if not choices:
            return pd.DataFrame()
        desired_rows = []
        for idx, c in enumerate(choices):
            desired_rows.append(
                {
                    "choice_id": idx,
                    "module_name": c.module_name,
                    "module_version": c.module_version,
                    "metric_id": c.metric_id,
                    "unit": c.unit,
                    "weighting": c.weighting,
                    "scope_id": c.scope_id,
                    "layer": c.layer,
                }
            )
        desired = pd.DataFrame(desired_rows)
        key_cols = ["module_name", "module_version", "metric_id", "unit", "weighting", "scope_id", "layer"]
        usecols = [
            "text_id", "text_version_id", "title", "author", "collection", "date_label", "genre",
            "module_name", "module_version", "metric_id", "value", "layer", "scope", "scope_id",
            "unit", "weighting", "denominator", "observation_count", "note",
        ]
        parts: list[pd.DataFrame] = []
        for chunk in self._iter_chunks(usecols=usecols):
            chunk = chunk.loc[chunk["scope"].eq("document")]
            merged = chunk.merge(desired, on=key_cols, how="inner", validate="many_to_many")
            if not merged.empty:
                parts.append(merged)
        if not parts:
            return pd.DataFrame()
        out = pd.concat(parts, ignore_index=True)
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        duplicate = out.duplicated(["choice_id", "text_id"], keep=False)
        if duplicate.any():
            sample = out.loc[duplicate, ["choice_id", "text_id", "metric_id", "scope_id", "weighting"]].head(10)
            raise VerseVADReaderError(
                "One exact module metric produced duplicate poem rows during broad scan:\n" + sample.to_string(index=False)
            )
        label_map = {i: c.label for i, c in enumerate(choices)}
        concept_map = {i: c.concept_label for i, c in enumerate(choices)}
        concept_key_map = {i: c.concept_key for i, c in enumerate(choices)}
        module_map = {i: c.module_name for i, c in enumerate(choices)}
        out["metric_label"] = out["choice_id"].map(label_map)
        out["concept_label"] = out["choice_id"].map(concept_map)
        out["concept_key"] = out["choice_id"].map(concept_key_map)
        out["module_name_for_score"] = out["choice_id"].map(module_map)
        out["coverage"] = np.nan
        return out

    def load_categorical_values(self, choices: Sequence[ModuleMetricChoice]) -> pd.DataFrame:
        if not choices:
            return pd.DataFrame()
        desired_rows = []
        for idx, c in enumerate(choices):
            desired_rows.append(
                {
                    "choice_id": idx,
                    "module_name": c.module_name,
                    "module_version": c.module_version,
                    "metric_id": c.metric_id,
                    "unit": c.unit,
                    "weighting": c.weighting,
                    "scope_id": c.scope_id,
                    "layer": c.layer,
                }
            )
        desired = pd.DataFrame(desired_rows)
        key_cols = ["module_name", "module_version", "metric_id", "unit", "weighting", "scope_id", "layer"]
        usecols = [
            "text_id", "text_version_id", "title", "author", "collection", "date_label", "genre",
            "module_name", "module_version", "metric_id", "value", "layer", "scope", "scope_id",
            "unit", "weighting", "note",
        ]
        parts: list[pd.DataFrame] = []
        for chunk in self._iter_chunks(usecols=usecols):
            chunk = chunk.loc[chunk["scope"].eq("document")]
            merged = chunk.merge(desired, on=key_cols, how="inner", validate="many_to_many")
            if not merged.empty:
                parts.append(merged)
        if not parts:
            return pd.DataFrame()
        out = pd.concat(parts, ignore_index=True)
        duplicate = out.duplicated(["choice_id", "text_id"], keep=False)
        if duplicate.any():
            return pd.DataFrame()  # safest: omit ambiguous categorical identities from broad scan
        out["metric_label"] = out["choice_id"].map({i: c.label for i, c in enumerate(choices)})
        out["module_name_for_score"] = out["choice_id"].map({i: c.module_name for i, c in enumerate(choices)})
        return out


def _module_profile_match(scope_id: str, weighting: str, scope: str, preferred_weighting: str) -> bool:
    """Include profile-independent metrics plus exact variants for preferred profile."""

    sid = (scope_id or "").strip()
    w = (weighting or "").strip()
    if not sid:
        return True

    direct = {
        ("all_matched", "token"): "all_token",
        ("all_matched", "type"): "all_type",
        ("stopwords_excluded", "token"): "stopwords_excluded_token",
        ("stopwords_excluded", "type"): "stopwords_excluded_type",
    }
    if sid in {"all_token", "all_type", "stopwords_excluded_token", "stopwords_excluded_type"}:
        return sid == direct.get((scope, preferred_weighting), "")

    # PoetryID and similar resource-qualified scope IDs, e.g. nrc_vad_v2_1:content_words.
    if ":" in sid:
        suffix = sid.rsplit(":", 1)[-1]
        if suffix in {"all_matched", "stopwords_excluded", "content_words"}:
            return suffix == scope and w == preferred_weighting

    # Unknown scoped variant: avoid silently mixing an unrelated profile.
    return False


def friendly_module_label(row: pd.Series) -> str:
    metric_id = str(row["metric_id"])
    module_name = str(row["module_name"])
    unit = str(row["unit"] or "")
    weighting = str(row["weighting"] or "")
    scope_id = str(row["scope_id"] or "")
    base = MODULE_METRIC_LABELS.get(metric_id)
    if not base:
        pieces = metric_id.split(".")
        # Remove redundant module-like prefixes for readability.
        if len(pieces) > 1:
            pieces = pieces[1:]
        base = _pretty_words(" ".join(pieces))
    module = MODULE_NAMES.get(module_name, _pretty_words(module_name))

    profile_note = ""
    if scope_id:
        if scope_id in {"all_token", "all_type", "stopwords_excluded_token", "stopwords_excluded_type"}:
            profile_note = {
                "all_token": " · All lexical tokens / token",
                "all_type": " · All lexical tokens / type",
                "stopwords_excluded_token": " · Stopword-excluded / token",
                "stopwords_excluded_type": " · Stopword-excluded / type",
            }[scope_id]
        elif ":" in scope_id:
            resource, lexical_scope = scope_id.rsplit(":", 1)
            resource = {
                "nrc_vad_v1": "NRC VAD v1",
                "nrc_vad_v2_1": "NRC VAD v2.1",
                "warriner_vad_2013": "Warriner VAD",
            }.get(resource, resource)
            profile_note = f" · {resource} / {VIEW_LABELS.get(lexical_scope, lexical_scope)} / {weighting or 'native'}"
    return f"{base} · {module}{profile_note}"


def module_metric_hard_excluded(metric_id: str) -> bool:
    mid = metric_id.casefold()
    return any(token in mid for token in MODULE_HARD_EXCLUDE_PATTERNS)


def available_module_choices(
    reader: ModuleMetricsReader,
    preferred_scope: str,
    preferred_weighting: str,
    *,
    numeric_only: bool = True,
) -> list[ModuleMetricChoice]:
    catalog = reader.catalog()
    if catalog.empty:
        return []
    choices: list[ModuleMetricChoice] = []
    for _, row in catalog.iterrows():
        if bool(row.get("duplicate_work", False)):
            continue
        metric_id = str(row["metric_id"])
        if module_metric_hard_excluded(metric_id):
            continue
        numeric = bool(row["numeric"])
        if numeric_only and not numeric:
            continue
        if not _module_profile_match(
            str(row["scope_id"]), str(row["weighting"]), preferred_scope, preferred_weighting
        ):
            continue
        choice = ModuleMetricChoice(
            label=friendly_module_label(row),
            module_name=str(row["module_name"]),
            module_version=str(row["module_version"]),
            metric_id=metric_id,
            unit=str(row["unit"]),
            weighting=str(row["weighting"]),
            scope_id=str(row["scope_id"]),
            layer=str(row["layer"]),
            note=str(row["note"]),
            numeric=numeric,
        )
        choices.append(choice)
    return sorted(choices, key=lambda c: (c.module_name, c.label.casefold()))


# ---------------------------------------------------------------------------
# Shared metric search UI
# ---------------------------------------------------------------------------


def _normalize_search(value: str) -> str:
    value = value.casefold().replace("acquisition", "aoa acquisition")
    value = re.sub(r"[^a-z0-9.]+", " ", value)
    return " ".join(value.split())


def search_choices(choices: Sequence[object], query: str) -> list[object]:
    terms = _normalize_search(query).split()
    if not terms:
        return []
    scored: list[tuple[int, str, object]] = []
    for choice in choices:
        if isinstance(choice, LexicalMetricChoice):
            fields = [
                choice.label, choice.lexicon_id, choice.lexicon, choice.metric,
                choice.dimension, choice.category, choice.scale,
            ]
        elif isinstance(choice, ModuleMetricChoice):
            fields = [
                choice.label, choice.module_name, choice.metric_id, choice.unit,
                choice.weighting, choice.scope_id, choice.note,
            ]
        else:
            continue
        haystack = _normalize_search(" ".join(fields))
        if all(term in haystack for term in terms):
            label_norm = _normalize_search(getattr(choice, "label"))
            concept_norm = _normalize_search(getattr(choice, "concept_label", getattr(choice, "label")))
            query_norm = _normalize_search(query)
            # Prefer an exact concept/label phrase match, then a contiguous phrase,
            # then ordinary term coverage. This keeps searches such as "rhyme density"
            # from being swamped by every density metric in the Rhyme module.
            exact_bonus = 3 if concept_norm == query_norm else 2 if label_norm == query_norm else 0
            phrase_bonus = 1 if query_norm in concept_norm else 0
            label_hits = sum(term in label_norm for term in terms)
            if isinstance(choice, LexicalMetricChoice):
                priority = _lexical_metric_priority(choice)[0]
            else:
                mid = getattr(choice, "metric_id", "").casefold()
                if mid.endswith(".mean") or "mean_" in mid or mid.endswith("_mean"):
                    priority = 0
                elif "standard_deviation" in mid or "variability" in mid or "interquartile_range" in mid:
                    priority = 1
                elif "proportion" in mid or "density" in mid or "score" in mid or "fit" in mid:
                    priority = 2
                elif "cumulative" in mid or "load" in mid:
                    priority = 3
                else:
                    priority = 4
            scored.append((-(exact_bonus * 100 + phrase_bonus * 20 + label_hits), priority, getattr(choice, "label").casefold(), choice))
    scored.sort(key=lambda t: (t[0], t[1], t[2]))
    return [x[3] for x in scored]


def select_metric_interactively(choices: Sequence[object], prompt_name: str = "metric") -> Optional[object]:
    while True:
        print(f"\nChoose {prompt_name}")
        print(
            "Search by ordinary terms such as: concreteness, frequency, interoceptive, "
            "arousal nrc v2, rhyme density, meter fit, rhythmic variability, readability, mtld."
        )
        print("Type Q to cancel.")
        query = input("Search: ").strip()
        if query.casefold() in {"q", "quit", "cancel"}:
            return None
        matches = search_choices(choices, query)
        if not matches:
            print("No matching metrics. Try a broader or different search term.")
            continue
        shown = matches[:MAX_SEARCH_RESULTS]
        print(f"\nMatches ({len(matches)} total):")
        for idx, choice in enumerate(shown, start=1):
            print(f"[{idx}] {getattr(choice, 'label')}")
            if isinstance(choice, LexicalMetricChoice):
                print(
                    f"    metric={choice.metric}; dimension={choice.dimension or '(none)'}; "
                    f"scale={choice.scale or '(unspecified)'}"
                )
            else:
                assert isinstance(choice, ModuleMetricChoice)
                print(
                    f"    metric_id={choice.metric_id}; unit={choice.unit or '(unspecified)'}; "
                    f"weighting={choice.weighting or 'module-native'}"
                )
        if len(matches) > len(shown):
            print(f"Showing the first {len(shown)} matches. Refine your search if needed.")
        while True:
            raw = input("Select number, R to refine search, or Q to cancel: ").strip().casefold()
            if raw in {"r", "refine"}:
                break
            if raw in {"q", "quit", "cancel"}:
                return None
            try:
                number = int(raw)
            except ValueError:
                print("Enter a displayed number, R, or Q.")
                continue
            if 1 <= number <= len(shown):
                return shown[number - 1]
            print(f"Choose a number from 1 to {len(shown)}.")


# ---------------------------------------------------------------------------
# Descriptive helpers
# ---------------------------------------------------------------------------


def percentile_ranks(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=values.index, dtype=float)
    mask = numeric.notna()
    n = int(mask.sum())
    if n == 0:
        return result
    ranks = rankdata(numeric.loc[mask].to_numpy(dtype=float), method="average")
    result.loc[mask] = (ranks - 0.5) / n * 100.0
    return result


def z_scores(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=values.index, dtype=float)
    mask = numeric.notna()
    if int(mask.sum()) == 0:
        return result
    arr = numeric.loc[mask].to_numpy(dtype=float)
    sd = float(np.std(arr, ddof=0))
    if not math.isfinite(sd) or sd == 0:
        result.loc[mask] = 0.0
        return result
    result.loc[mask] = (arr - float(np.mean(arr))) / sd
    return result


def percentile_phrase(percentile: float, direction: Optional[str] = None) -> str:
    if not math.isfinite(percentile):
        return "percentile unavailable"
    if direction is None:
        direction = "high" if percentile >= 50 else "low"
    directional = percentile if direction == "high" else 100.0 - percentile
    if directional >= 97.5:
        adjective = "extremely"
    elif directional >= 90:
        adjective = "very"
    elif directional >= 75:
        adjective = "unusually"
    else:
        adjective = "somewhat"
    return f"{adjective} {'high' if direction == 'high' else 'low'} (percentile {percentile:.1f})"


def apply_coverage_filter(df: pd.DataFrame, threshold: Optional[float]) -> pd.DataFrame:
    out = df.loc[df["value"].notna()].copy()
    if threshold is not None:
        out = out.loc[out["coverage"].notna() & (out["coverage"] >= threshold)].copy()
    return out


# ---------------------------------------------------------------------------
# Lexical analyses
# ---------------------------------------------------------------------------


def prepare_lexical_single(
    reader: VerseVADCorpusReader, choice: LexicalMetricChoice, coverage_threshold: Optional[float]
) -> pd.DataFrame:
    data = apply_coverage_filter(reader.select_metric(choice.to_spec()).copy(), coverage_threshold)
    if data.empty:
        raise VerseVADReaderError(f"No eligible poems remain for {choice.label} after filtering.")
    data["percentile_rank"] = percentile_ranks(data["value"])
    data["z_score"] = z_scores(data["value"])
    return data


def analyze_lexical_single(
    reader: VerseVADCorpusReader,
    choice: LexicalMetricChoice,
    direction: str,
    result_count: int,
    coverage_threshold: Optional[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = prepare_lexical_single(reader, choice, coverage_threshold)
    return build_single_results(data, choice.label, choice.concept_label, direction, result_count)


def analyze_lexical_pair(
    reader: VerseVADCorpusReader,
    a: LexicalMetricChoice,
    b: LexicalMetricChoice,
    direction_a: str,
    direction_b: str,
    extremity_threshold: float,
    result_count: int,
    coverage_threshold: Optional[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired = reader.pair_metrics(a.to_spec(), b.to_spec(), drop_missing=False).copy()
    paired = paired.loc[paired["complete_pair"]].copy()
    if coverage_threshold is not None:
        paired = paired.loc[
            paired["x_coverage"].notna() & paired["y_coverage"].notna()
            & (paired["x_coverage"] >= coverage_threshold)
            & (paired["y_coverage"] >= coverage_threshold)
        ].copy()
    return build_pair_results(
        paired, a.label, b.label, a.concept_label, b.concept_label,
        direction_a, direction_b, extremity_threshold, result_count,
    )


def batch_select_lexical(reader: VerseVADCorpusReader, choices: Sequence[LexicalMetricChoice]) -> pd.DataFrame:
    if not choices:
        return pd.DataFrame()
    desired = pd.DataFrame(
        [
            {
                "choice_id": i,
                "lexicon_id": c.lexicon_id,
                "metric": c.metric,
                "dimension": c.dimension,
                "category": c.category,
                "analysis_view": c.analysis_view,
                "weighting": c.weighting,
            }
            for i, c in enumerate(choices)
        ]
    )
    keys = ["lexicon_id", "metric", "dimension", "category", "analysis_view", "weighting"]
    usecols = [
        "text_id", "text_version_id", "title", "author", "collection", "date_label", "genre",
        "lexicon_id", "lexicon", "metric", "dimension", "category", "analysis_view", "weighting",
        "scale", "denominator", "value", "observations", "matched_tokens", "lexical_tokens", "coverage",
    ]
    parts: list[pd.DataFrame] = []
    for chunk in reader._iter_chunks(usecols=usecols):  # type: ignore[attr-defined]
        merged = chunk.merge(desired, on=keys, how="inner", validate="many_to_many")
        if not merged.empty:
            parts.append(merged)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    for col in ["value", "observations", "matched_tokens", "lexical_tokens", "coverage"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    duplicate = out.duplicated(["choice_id", "text_id"], keep=False)
    if duplicate.any():
        raise VerseVADReaderError("One exact lexical metric produced duplicate poem rows during broad scan.")
    out["metric_label"] = out["choice_id"].map({i: c.label for i, c in enumerate(choices)})
    out["concept_label"] = out["choice_id"].map({i: c.concept_label for i, c in enumerate(choices)})
    out["resource_label"] = out["choice_id"].map({i: c.resource_label for i, c in enumerate(choices)})
    return out


def analyze_lexical_general(
    reader: VerseVADCorpusReader,
    choices: Sequence[LexicalMetricChoice],
    result_count: int,
    coverage_threshold: Optional[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[LexicalMetricChoice]]:
    scan_choices = [c for c in choices if is_lexical_general_metric(c.metric)]
    if not scan_choices:
        raise VerseVADReaderError("No eligible lexical metrics are available for the broad scan.")
    long = batch_select_lexical(reader, scan_choices)
    long = long.loc[long["value"].notna()].copy()
    if coverage_threshold is not None:
        long = long.loc[long["coverage"].notna() & (long["coverage"] >= coverage_threshold)].copy()

    processed: list[pd.DataFrame] = []
    retained_ids: set[int] = set()
    for choice_id, group in long.groupby("choice_id", sort=False):
        if len(group) < GENERAL_MIN_METRIC_N:
            continue
        g = group.copy()
        g["metric_percentile"] = percentile_ranks(g["value"])
        g["metric_z_score"] = z_scores(g["value"])
        g["metric_extremeness"] = 2.0 * (g["metric_percentile"] - 50.0).abs()
        processed.append(g)
        retained_ids.add(int(choice_id))
    if not processed:
        raise VerseVADReaderError("No broad-scan metrics retained enough eligible poems.")
    evidence = pd.concat(processed, ignore_index=True)

    concept = (
        evidence.groupby(
            ["text_id", "text_version_id", "title", "author", "collection", "date_label", "genre", "concept_label"],
            dropna=False, as_index=False,
        )
        .agg(
            consensus_percentile=("metric_percentile", "median"),
            consensus_z_score=("metric_z_score", "median"),
            resource_count=("resource_label", "nunique"),
            resources=("resource_label", lambda s: "; ".join(sorted(set(map(str, s))))),
            min_coverage=("coverage", "min"),
            mean_coverage=("coverage", "mean"),
        )
    )
    concept["concept_extremeness"] = 2.0 * (concept["consensus_percentile"] - 50.0).abs()
    concept["direction"] = np.where(concept["consensus_percentile"] >= 50.0, "high", "low")

    score_rows: list[dict[str, object]] = []
    for text_id, group in concept.groupby("text_id", sort=False):
        g = group.sort_values(["concept_extremeness", "concept_label"], ascending=[False, True]).copy()
        if len(g) < GENERAL_MIN_CONCEPTS:
            continue
        top = g.head(GENERAL_TOP_COMPONENTS)
        pieces = [
            f"{row['concept_label']} is {percentile_phrase(float(row['consensus_percentile']), str(row['direction']))}"
            + (f" across {int(row['resource_count'])} resources" if int(row["resource_count"]) > 1 else "")
            for _, row in top.iterrows()
        ]
        first = g.iloc[0]
        score = float(g["concept_extremeness"].mean())
        top3_score = float(top["concept_extremeness"].mean())
        score_rows.append(
            {
                "text_id": text_id,
                "text_version_id": first["text_version_id"],
                "title": first["title"],
                "author": first["author"],
                "collection": first["collection"],
                "date_label": first["date_label"],
                "genre": first["genre"],
                "anomaly_type": "broad lexical profile",
                "overall_anomaly_score": score,
                "top_three_extremeness_score": top3_score,
                "maximum_single_concept_extremeness": float(g["concept_extremeness"].max()),
                "extreme_concept_count": int((g["concept_extremeness"] >= 80).sum()),
                "concepts_available": len(g),
                "description": "; ".join(pieces) + ".",
            }
        )
    scores = pd.DataFrame(score_rows)
    if scores.empty:
        raise VerseVADReaderError("No poems had enough eligible concepts for the broad anomaly score.")
    scores["corpus_anomaly_percentile"] = percentile_ranks(scores["overall_anomaly_score"])
    scores = scores.sort_values(
        ["overall_anomaly_score", "top_three_extremeness_score", "title"],
        ascending=[False, False, True],
    ).copy()
    scores.insert(0, "anomaly_rank", range(1, len(scores) + 1))
    results = scores.head(result_count).copy()
    top_ids = set(results["text_id"])
    top_evidence = concept.loc[concept["text_id"].isin(top_ids)].merge(
        results[["text_id", "anomaly_rank"]], on="text_id", how="left"
    ).sort_values(["anomaly_rank", "concept_extremeness"], ascending=[True, False])
    retained = [c for i, c in enumerate(scan_choices) if i in retained_ids]
    return results, scores, top_evidence, retained


# ---------------------------------------------------------------------------
# Module analyses
# ---------------------------------------------------------------------------


def build_single_results(
    data: pd.DataFrame,
    metric_label: str,
    concept_label: str,
    direction: str,
    result_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    if direction in {"high", "both"}:
        high = data.sort_values(["value", "title"], ascending=[False, True]).head(result_count).copy()
        high.insert(0, "result_direction", "highest")
        high.insert(0, "result_rank", range(1, len(high) + 1))
        frames.append(high)
    if direction in {"low", "both"}:
        low = data.sort_values(["value", "title"], ascending=[True, True]).head(result_count).copy()
        low.insert(0, "result_direction", "lowest")
        low.insert(0, "result_rank", range(1, len(low) + 1))
        frames.append(low)
    results = pd.concat(frames, ignore_index=True)
    results["metric_label"] = metric_label
    results["description"] = results.apply(
        lambda row: (
            f"{row['result_direction'].capitalize()} {concept_label}: value {float(row['value']):.6g}, "
            f"corpus percentile {float(row['percentile_rank']):.1f}."
        ),
        axis=1,
    )
    all_ranked = data.sort_values(["value", "title"], ascending=[False, True]).copy()
    all_ranked.insert(0, "descending_rank", range(1, len(all_ranked) + 1))
    return results, all_ranked


def build_pair_results(
    paired: pd.DataFrame,
    a_label: str,
    b_label: str,
    a_concept: str,
    b_concept: str,
    direction_a: str,
    direction_b: str,
    extremity_threshold: float,
    result_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if paired.empty:
        raise VerseVADReaderError("No eligible paired poems remain after filtering.")
    paired["a_percentile"] = percentile_ranks(paired["x_value"])
    paired["b_percentile"] = percentile_ranks(paired["y_value"])
    paired["a_z_score"] = z_scores(paired["x_value"])
    paired["b_z_score"] = z_scores(paired["y_value"])
    paired["a_directional_extremity"] = paired["a_percentile"] if direction_a == "high" else 100 - paired["a_percentile"]
    paired["b_directional_extremity"] = paired["b_percentile"] if direction_b == "high" else 100 - paired["b_percentile"]
    paired["weakest_directional_extremity"] = paired[["a_directional_extremity", "b_directional_extremity"]].min(axis=1)
    paired["joint_directional_score"] = np.sqrt(
        paired["a_directional_extremity"] * paired["b_directional_extremity"]
    )
    paired["meets_threshold"] = (
        (paired["a_directional_extremity"] >= extremity_threshold)
        & (paired["b_directional_extremity"] >= extremity_threshold)
    )
    paired["metric_a_label"] = a_label
    paired["metric_b_label"] = b_label
    paired["metric_a_direction"] = direction_a
    paired["metric_b_direction"] = direction_b
    paired["description"] = paired.apply(
        lambda row: (
            f"{a_concept} is {percentile_phrase(float(row['a_percentile']), direction_a)}; "
            f"{b_concept} is {percentile_phrase(float(row['b_percentile']), direction_b)}."
        ), axis=1,
    )
    ranked = paired.sort_values(
        ["joint_directional_score", "weakest_directional_extremity", "title"],
        ascending=[False, False, True],
    ).copy()
    ranked.insert(0, "directional_rank", range(1, len(ranked) + 1))
    return ranked.loc[ranked["meets_threshold"]].head(result_count).copy(), ranked


def prepare_module_single(reader: ModuleMetricsReader, choice: ModuleMetricChoice) -> pd.DataFrame:
    data = reader.select_metric(choice).copy()
    data = data.loc[data["value"].notna()].copy()
    if data.empty:
        raise VerseVADReaderError(f"No numeric poem values remain for {choice.label}.")
    data["percentile_rank"] = percentile_ranks(data["value"])
    data["z_score"] = z_scores(data["value"])
    return data


def analyze_module_single(
    reader: ModuleMetricsReader,
    choice: ModuleMetricChoice,
    direction: str,
    result_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return build_single_results(
        prepare_module_single(reader, choice), choice.label, choice.concept_label, direction, result_count
    )


def analyze_module_pair(
    reader: ModuleMetricsReader,
    a: ModuleMetricChoice,
    b: ModuleMetricChoice,
    direction_a: str,
    direction_b: str,
    extremity_threshold: float,
    result_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired = reader.pair_metrics(a, b)
    paired = paired.loc[paired["complete_pair"]].copy()
    return build_pair_results(
        paired, a.label, b.label, a.concept_label, b.concept_label,
        direction_a, direction_b, extremity_threshold, result_count,
    )


def module_general_numeric_eligible(choice: ModuleMetricChoice, include_size_metrics: bool) -> bool:
    mid = choice.metric_id.casefold()
    if any(token in mid for token in MODULE_GENERAL_EXCLUDE_PATTERNS):
        return False
    if choice.module_name == "versemap":
        # VerseMap duplicates many already-available metrics and includes technical coordinates.
        return False
    if not include_size_metrics and any(token in mid for token in MODULE_SIZE_PATTERNS):
        return False
    return choice.numeric


def module_general_categorical_choices(
    reader: ModuleMetricsReader,
    preferred_scope: str,
    preferred_weighting: str,
) -> list[ModuleMetricChoice]:
    catalog = reader.catalog()
    choices: list[ModuleMetricChoice] = []
    for _, row in catalog.iterrows():
        if bool(row.get("duplicate_work", False)) or bool(row["numeric"]):
            continue
        metric_id = str(row["metric_id"])
        if module_metric_hard_excluded(metric_id):
            continue
        if any(token in metric_id.casefold() for token in ("_id", ".id", "result_status", "profile_id", "confidence")):
            # Confidence/evidence labels describe analysis support, not the poem's
            # substantive stylistic/formal profile, so they are not anomaly targets.
            continue
        unique = int(row["unique_values"])
        if not (2 <= unique <= MODULE_RARE_CATEGORY_MAX_LEVELS):
            continue
        if not _module_profile_match(
            str(row["scope_id"]), str(row["weighting"]), preferred_scope, preferred_weighting
        ):
            continue
        choices.append(
            ModuleMetricChoice(
                label=friendly_module_label(row),
                module_name=str(row["module_name"]),
                module_version=str(row["module_version"]),
                metric_id=metric_id,
                unit=str(row["unit"]),
                weighting=str(row["weighting"]),
                scope_id=str(row["scope_id"]),
                layer=str(row["layer"]),
                note=str(row["note"]),
                numeric=False,
            )
        )
    return choices


def analyze_module_general(
    reader: ModuleMetricsReader,
    numeric_choices: Sequence[ModuleMetricChoice],
    preferred_scope: str,
    preferred_weighting: str,
    result_count: int,
    include_size_metrics: bool,
    include_rare_categories: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[ModuleMetricChoice], list[ModuleMetricChoice]]:
    scan_choices = [c for c in numeric_choices if module_general_numeric_eligible(c, include_size_metrics)]
    if not scan_choices:
        raise VerseVADReaderError("No eligible numeric module metrics are available for broad scan.")
    long = reader.batch_select_numeric(scan_choices)
    long = long.loc[long["value"].notna()].copy()

    processed: list[pd.DataFrame] = []
    retained_ids: set[int] = set()
    for choice_id, group in long.groupby("choice_id", sort=False):
        if len(group) < GENERAL_MIN_METRIC_N or group["value"].nunique(dropna=True) < 2:
            continue
        g = group.copy()
        g["metric_percentile"] = percentile_ranks(g["value"])
        g["metric_z_score"] = z_scores(g["value"])
        g["metric_extremeness"] = 2.0 * (g["metric_percentile"] - 50.0).abs()
        g["direction"] = np.where(g["metric_percentile"] >= 50.0, "high", "low")
        processed.append(g)
        retained_ids.add(int(choice_id))
    if not processed:
        raise VerseVADReaderError("No module metrics retained enough eligible poems for broad scan.")
    evidence = pd.concat(processed, ignore_index=True)

    # Collapse parallel resource/profile identities of the same module metric.
    concept = (
        evidence.groupby(
            ["text_id", "text_version_id", "title", "author", "collection", "date_label", "genre",
             "module_name_for_score", "concept_key", "concept_label"],
            dropna=False, as_index=False,
        )
        .agg(
            consensus_percentile=("metric_percentile", "median"),
            consensus_z_score=("metric_z_score", "median"),
            identity_count=("choice_id", "nunique"),
        )
    )
    concept["concept_extremeness"] = 2.0 * (concept["consensus_percentile"] - 50.0).abs()
    concept["direction"] = np.where(concept["consensus_percentile"] >= 50.0, "high", "low")

    # Balance the broad profile by module so a module with many exported metrics
    # cannot dominate simply because it contributes more columns.
    module_scores = (
        concept.groupby(["text_id", "module_name_for_score"], as_index=False)
        .agg(module_extremeness=("concept_extremeness", "mean"), concepts_in_module=("concept_key", "nunique"))
    )
    profile = (
        module_scores.groupby("text_id", as_index=False)
        .agg(
            broad_profile_score=("module_extremeness", "mean"),
            modules_available=("module_name_for_score", "nunique"),
        )
    )

    top_numeric = (
        concept.sort_values(["text_id", "concept_extremeness"], ascending=[True, False])
        .groupby("text_id", sort=False)
        .head(GENERAL_TOP_COMPONENTS)
        .copy()
    )
    top_numeric_score = (
        top_numeric.groupby("text_id", as_index=False)
        .agg(
            top_numeric_extremeness=("concept_extremeness", "mean"),
            strongest_numeric_extremeness=("concept_extremeness", "max"),
            extreme_numeric_count=("concept_extremeness", lambda s: int((s >= 80).sum())),
        )
    )
    profile = profile.merge(top_numeric_score, on="text_id", how="left")

    # Optional rare categorical labels. We intentionally use only interpretable
    # low-cardinality categories, never IDs/hashes or huge rhyme-scheme vocabularies.
    categorical_choices: list[ModuleMetricChoice] = []
    categorical_evidence = pd.DataFrame()
    if include_rare_categories:
        categorical_choices = module_general_categorical_choices(reader, preferred_scope, preferred_weighting)
        categorical = reader.load_categorical_values(categorical_choices)
        cat_parts: list[pd.DataFrame] = []
        if not categorical.empty:
            for choice_id, g in categorical.groupby("choice_id", sort=False):
                nonblank = g.loc[g["value"].astype(str).str.strip().ne("")].copy()
                n = len(nonblank)
                if n < GENERAL_MIN_METRIC_N:
                    continue
                counts = nonblank["value"].value_counts()
                nonblank["category_count"] = nonblank["value"].map(counts)
                nonblank["category_share"] = nonblank["category_count"] / n
                nonblank["category_rarity_score"] = (1.0 - nonblank["category_share"]) * 100.0
                rare = nonblank.loc[nonblank["category_share"] <= MODULE_RARE_CATEGORY_MAX_SHARE].copy()
                if not rare.empty:
                    cat_parts.append(rare)
        if cat_parts:
            categorical_evidence = pd.concat(cat_parts, ignore_index=True)

    strongest_cat = pd.DataFrame(columns=["text_id", "strongest_categorical_rarity"])
    if not categorical_evidence.empty:
        strongest_cat = (
            categorical_evidence.groupby("text_id", as_index=False)
            .agg(strongest_categorical_rarity=("category_rarity_score", "max"))
        )
    profile = profile.merge(strongest_cat, on="text_id", how="left")
    profile["strongest_categorical_rarity"] = pd.to_numeric(
        profile["strongest_categorical_rarity"], errors="coerce"
    ).fillna(0.0)
    # The primary broad score remains module-balanced numeric extremeness. Rare
    # categories are reported in their own ranked evidence table rather than
    # being mixed into one arbitrary omnibus score.
    profile["overall_anomaly_score"] = profile["broad_profile_score"]

    # Attach metadata and build human-readable descriptions.
    metadata = concept[
        ["text_id", "text_version_id", "title", "author", "collection", "date_label", "genre"]
    ].drop_duplicates("text_id")
    scores = profile.merge(metadata, on="text_id", how="left")

    descriptions: dict[str, str] = {}
    types: dict[str, str] = {}
    for _, row in scores.iterrows():
        text_id = str(row["text_id"])
        numeric_top = top_numeric.loc[top_numeric["text_id"].eq(text_id)].sort_values(
            "concept_extremeness", ascending=False
        )
        pieces: list[str] = []
        for _, ev in numeric_top.iterrows():
            pieces.append(
                f"{ev['concept_label']} is {percentile_phrase(float(ev['consensus_percentile']), str(ev['direction']))}"
            )
        cat_piece = None
        if not categorical_evidence.empty:
            cats = categorical_evidence.loc[categorical_evidence["text_id"].eq(text_id)].sort_values(
                "category_rarity_score", ascending=False
            )
            if not cats.empty:
                ev = cats.iloc[0]
                cat_piece = (
                    f"rare {ev['metric_label']} category '{ev['value']}' "
                    f"({int(ev['category_count'])} of {int(round(ev['category_count'] / ev['category_share']))} poems)"
                )
                pieces.append(cat_piece)

        directions = set(numeric_top["direction"].astype(str))
        if directions == {"high", "low"}:
            anomaly_type = "cross-dimensional contrast"
        elif directions == {"high"}:
            anomaly_type = "multi-metric high profile"
        elif directions == {"low"}:
            anomaly_type = "multi-metric low profile"
        else:
            anomaly_type = "broad multi-module profile"
        types[text_id] = anomaly_type
        descriptions[text_id] = "; ".join(pieces[: GENERAL_TOP_COMPONENTS + (1 if cat_piece else 0)]) + "."

    scores["anomaly_type"] = scores["text_id"].map(types)
    scores["description"] = scores["text_id"].map(descriptions)
    scores = scores.loc[scores["modules_available"] >= MODULE_GENERAL_MIN_MODULES].copy()
    scores["corpus_anomaly_percentile"] = percentile_ranks(scores["overall_anomaly_score"])
    scores = scores.sort_values(
        ["overall_anomaly_score", "top_numeric_extremeness", "strongest_categorical_rarity", "title"],
        ascending=[False, False, False, True],
    ).copy()
    scores.insert(0, "anomaly_rank", range(1, len(scores) + 1))
    results = scores.head(result_count).copy()

    top_ids = set(results["text_id"])
    numeric_ev = concept.loc[concept["text_id"].isin(top_ids)].copy()
    numeric_ev["evidence_type"] = "numeric"
    numeric_ev["evidence_detail"] = numeric_ev.apply(
        lambda r: f"{r['concept_label']} {percentile_phrase(float(r['consensus_percentile']), str(r['direction']))}",
        axis=1,
    )
    evidence_frames = [numeric_ev]
    if not categorical_evidence.empty:
        cat_ev = categorical_evidence.loc[categorical_evidence["text_id"].isin(top_ids)].copy()
        cat_ev["evidence_type"] = "categorical"
        cat_ev["evidence_detail"] = cat_ev.apply(
            lambda r: f"{r['metric_label']}='{r['value']}' share={float(r['category_share']):.3f}", axis=1
        )
        evidence_frames.append(cat_ev)
    combined_evidence = pd.concat(evidence_frames, ignore_index=True, sort=False)
    combined_evidence = combined_evidence.merge(
        results[["text_id", "anomaly_rank"]], on="text_id", how="left"
    ).sort_values(["anomaly_rank", "evidence_type"])

    rare_categories = pd.DataFrame()
    if not categorical_evidence.empty:
        rare_categories = categorical_evidence.sort_values(
            ["category_rarity_score", "category_share", "metric_label", "title"],
            ascending=[False, True, True, True],
        ).copy()
        rare_categories.insert(0, "rarity_rank", range(1, len(rare_categories) + 1))

    retained_numeric = [c for i, c in enumerate(scan_choices) if i in retained_ids]
    return results, scores, combined_evidence, rare_categories, retained_numeric, categorical_choices


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def slugify(value: str, max_length: int = 72) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return (value or "analysis")[:max_length].rstrip("_")


def source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataframe_rows(df: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> list[list[object]]:
    columns = list(columns) if columns is not None else list(df.columns)
    rows: list[list[object]] = [columns]
    for row in df.loc[:, columns].itertuples(index=False, name=None):
        clean: list[object] = []
        for value in row:
            if pd.isna(value):
                clean.append(None)
            elif isinstance(value, np.generic):
                clean.append(value.item())
            else:
                clean.append(value)
        rows.append(clean)
    return rows


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _clean_xml_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = "".join(ch for ch in text if ch in "\t\n\r" or ord(ch) >= 32)
    return xml_escape(text, {'"': "&quot;"})


def _excel_col(number: int) -> str:
    result = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _worksheet_xml(rows: Sequence[Sequence[object]]) -> str:
    xml_rows: list[str] = []
    for r_idx, row in enumerate(rows, start=1):
        cells: list[str] = []
        for c_idx, value in enumerate(row, start=1):
            if value is None:
                continue
            ref = f"{_excel_col(c_idx)}{r_idx}"
            style = ' s="1"' if r_idx == 1 else ""
            if isinstance(value, (bool, np.bool_)):
                cells.append(f'<c r="{ref}" t="b"{style}><v>{1 if bool(value) else 0}</v></c>')
            elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                numeric = float(value)
                if math.isfinite(numeric):
                    cells.append(f'<c r="{ref}"{style}><v>{numeric!r}</v></c>')
            else:
                text = _clean_xml_text(value)
                preserve = ' xml:space="preserve"' if text[:1].isspace() or text[-1:].isspace() else ""
                cells.append(f'<c r="{ref}" t="inlineStr"{style}><is><t{preserve}>{text}</t></is></c>')
        xml_rows.append(f'<row r="{r_idx}">{"".join(cells)}</row>')
    max_cols = max((len(row) for row in rows), default=1)
    max_rows = max(len(rows), 1)
    filter_ref = f"A1:{_excel_col(max_cols)}{max_rows}"
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        '<sheetFormatPr defaultRowHeight="15"/>'
        f'<sheetData>{"".join(xml_rows)}</sheetData>'
        f'<autoFilter ref="{filter_ref}"/>'
        '</worksheet>'
    )


def _styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2"><font><sz val="11"/><name val="Calibri"/><family val="2"/></font>'
        '<font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/><family val="2"/></font></fonts>'
        '<fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1" applyAlignment="1"><alignment wrapText="1"/></xf></cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>'
    )


def _sanitize_sheet_name(name: str, used: set[str]) -> str:
    clean = re.sub(r"[\\/*?:\[\]]", "", name).strip() or "Sheet"
    clean = clean[:31]
    candidate = clean
    number = 2
    while candidate.casefold() in used:
        suffix = f" {number}"
        candidate = (clean[: 31 - len(suffix)] + suffix).strip()
        number += 1
    used.add(candidate.casefold())
    return candidate


def write_simple_xlsx(path: Path, sheets: Sequence[tuple[str, list[list[object]]]]) -> None:
    if not sheets:
        raise ValueError("At least one worksheet is required.")
    used: set[str] = set()
    normalized = [(_sanitize_sheet_name(name, used), rows) for name, rows in sheets]
    overrides = "".join(
        f'<Override PartName="/xl/worksheets/sheet{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for idx in range(1, len(normalized) + 1)
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        f'{overrides}</Types>'
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    sheets_xml = "".join(
        f'<sheet name="{_clean_xml_text(name)}" sheetId="{idx}" r:id="rId{idx}"/>'
        for idx, (name, _) in enumerate(normalized, start=1)
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{sheets_xml}</sheets></workbook>'
    )
    rels = "".join(
        f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>'
        for idx in range(1, len(normalized) + 1)
    )
    rels += f'<Relationship Id="rId{len(normalized)+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'{rels}</Relationships>'
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/styles.xml", _styles_xml())
        for idx, (_, rows) in enumerate(normalized, start=1):
            archive.writestr(f"xl/worksheets/sheet{idx}.xml", _worksheet_xml(rows))


def metadata_rows(metadata: dict[str, object]) -> list[list[object]]:
    rows: list[list[object]] = [["field", "value"]]
    for key, value in metadata.items():
        rendered = json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
        rows.append([key, rendered])
    return rows


def create_run_folder(output_root: Path, source: Path, table_key: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = output_root / f"{slugify(source.stem)}_{table_key}_{stamp}"
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def export_single_mode(run_folder: Path, results: pd.DataFrame, all_ranked: pd.DataFrame, metadata: dict[str, object]) -> None:
    results.to_csv(run_folder / "anomaly_results.csv", index=False, encoding="utf-8-sig")
    all_ranked.to_csv(run_folder / "single_metric_all_ranked.csv", index=False, encoding="utf-8-sig")
    write_simple_xlsx(
        run_folder / "anomaly_analysis.xlsx",
        [("Results", dataframe_rows(results)), ("All Ranked", dataframe_rows(all_ranked)), ("Run Metadata", metadata_rows(metadata))],
    )


def export_pair_mode(run_folder: Path, results: pd.DataFrame, ranked: pd.DataFrame, metadata: dict[str, object]) -> None:
    results.to_csv(run_folder / "anomaly_results.csv", index=False, encoding="utf-8-sig")
    ranked.to_csv(run_folder / "pair_directional_all_ranked.csv", index=False, encoding="utf-8-sig")
    write_simple_xlsx(
        run_folder / "anomaly_analysis.xlsx",
        [("Threshold Matches", dataframe_rows(results)), ("All Ranked", dataframe_rows(ranked)), ("Run Metadata", metadata_rows(metadata))],
    )


def export_general_mode(
    run_folder: Path,
    results: pd.DataFrame,
    scores: pd.DataFrame,
    evidence: pd.DataFrame,
    metadata: dict[str, object],
    rare_categories: Optional[pd.DataFrame] = None,
) -> None:
    results.to_csv(run_folder / "anomaly_results.csv", index=False, encoding="utf-8-sig")
    scores.to_csv(run_folder / "general_anomaly_all_scores.csv", index=False, encoding="utf-8-sig")
    evidence.to_csv(run_folder / "general_anomaly_evidence.csv", index=False, encoding="utf-8-sig")
    sheets: list[tuple[str, list[list[object]]]] = [
        ("Top Anomalies", dataframe_rows(results)),
        ("All Scores", dataframe_rows(scores)),
        ("Evidence", dataframe_rows(evidence)),
    ]
    if rare_categories is not None and not rare_categories.empty:
        rare_categories.to_csv(
            run_folder / "general_rare_categories.csv", index=False, encoding="utf-8-sig"
        )
        sheets.append(("Rare Categories", dataframe_rows(rare_categories)))
    sheets.append(("Run Metadata", metadata_rows(metadata)))
    write_simple_xlsx(run_folder / "anomaly_analysis.xlsx", sheets)


# ---------------------------------------------------------------------------
# Terminal reporting
# ---------------------------------------------------------------------------


def print_single_results(results: pd.DataFrame, concept_label: str, show_coverage: bool) -> None:
    print("\nAnomaly Results")
    print("===============")
    current = None
    for _, row in results.iterrows():
        direction = str(row["result_direction"])
        if direction != current:
            current = direction
            print(f"\n{direction.capitalize()} {concept_label}")
        line = (
            f"{int(row['result_rank']):>2}. {row['title']} | value={float(row['value']):.4f} "
            f"| percentile={float(row['percentile_rank']):.1f}"
        )
        if show_coverage and "coverage" in row and pd.notna(row["coverage"]):
            line += f" | coverage={float(row['coverage']) * 100:.1f}%"
        print(line)


def print_pair_results(
    matches: pd.DataFrame,
    ranked: pd.DataFrame,
    a_concept: str,
    b_concept: str,
    direction_a: str,
    direction_b: str,
    threshold: float,
    result_count: int,
) -> None:
    print("\nAnomaly Results")
    print("===============")
    print(f"Requested combination: {direction_a.upper()} {a_concept} + {direction_b.upper()} {b_concept}")
    print(f"Both metrics must meet directional extremity ≥ {threshold:.1f} percentile points.")
    if matches.empty:
        print("\nNo poems met the strict two-metric threshold. Closest candidates:")
        shown = ranked.head(result_count)
    else:
        print(f"\n{len(matches)} matching poem(s) shown:")
        shown = matches
    for _, row in shown.iterrows():
        status = "MATCH" if bool(row["meets_threshold"]) else "near-match"
        print(f"\n{int(row['directional_rank']):>2}. {row['title']} [{status}]")
        print(f"    {a_concept}: {float(row['x_value']):.4f} (pct {float(row['a_percentile']):.1f})")
        print(f"    {b_concept}: {float(row['y_value']):.4f} (pct {float(row['b_percentile']):.1f})")
        print(f"    joint directional score: {float(row['joint_directional_score']):.1f}")


def print_general_results(results: pd.DataFrame, table_key: str) -> None:
    print("\nBroad Anomaly Results")
    print("=====================")
    if table_key == "vad":
        print("Corpus-relative lexical-profile anomalies. These are descriptive, not significance tests.")
    else:
        print("Corpus-relative multi-module anomalies. Numeric profile scores are module-balanced;")
        print("rare low-cardinality form/meter labels may also be surfaced. These are descriptive, not significance tests.")
    for _, row in results.iterrows():
        score_col = "overall_anomaly_score"
        print(f"\n{int(row['anomaly_rank']):>2}. {row['title']} | score={float(row[score_col]):.1f} | {row['anomaly_type']}")
        print(f"    {row['description']}")


# ---------------------------------------------------------------------------
# CLI main
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive exploratory anomaly finder for VerseVAD corpus exports.")
    parser.add_argument("--source", help="Optional source ZIP/CSV path. If omitted, choose interactively from source/.")
    parser.add_argument("--table", choices=["vad", "module"], help="Optional evidence table override.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_console_encoding()
    parser = _build_parser()
    args = parser.parse_args(argv)
    root = project_root()
    sources_dir = source_directory(root)
    output_root = export_directory(root)

    print("VerseVAD Anomaly Finder")
    print("========================\n")

    try:
        source = Path(args.source).expanduser().resolve() if args.source else choose_source_interactively(discover_sources(sources_dir))
        if not source.exists():
            raise FileNotFoundError(source)
        tables = available_tables(source)
        table_key = choose_table_interactively(source, tables, args.table)

        if table_key == "vad":
            print("\nValidating lexical / normative corpus table...")
            reader = VerseVADCorpusReader(source)
            report = reader.validate()
            print(
                f"Validated: {report.work_count:,} works · {report.lexicon_count} lexical resources · "
                f"schema {report.schema_fingerprint}"
            )
            analysis_view, weighting = choose_lexical_profile()
            print("\nBuilding available metric catalog for this profile...")
            choices = available_lexical_choices(reader, analysis_view, weighting)
            print(f"Available metric identities: {len(choices)}")
            coverage_threshold = prompt_coverage_threshold()
            profile_label = f"{VIEW_LABELS[analysis_view]} · {WEIGHT_LABELS[weighting]}"
            schema_fingerprint = report.schema_fingerprint
            corpus_work_count = report.work_count
            table_filename = VAD_METRICS_FILENAME
        else:
            print("\nValidating full module metrics table...")
            module_reader = ModuleMetricsReader(source)
            module_report = module_reader.validate()
            print(
                f"Validated: {module_report.work_count:,} works · {module_report.module_count} modules · "
                f"{module_report.metric_id_count} metric IDs · schema {module_report.schema_fingerprint}"
            )
            analysis_view, weighting = choose_module_profile()
            print("\nBuilding numeric document-level module metric catalog...")
            module_choices = available_module_choices(module_reader, analysis_view, weighting, numeric_only=True)
            print(f"Available numeric metric identities: {len(module_choices)}")
            print("Coverage threshold: not applied in full-module mode because rhyme, meter, readability,")
            print("and lexical modules do not share one comparable poem-level coverage denominator.")
            coverage_threshold = None
            profile_label = f"Preferred profile where applicable: {VIEW_LABELS[analysis_view]} · {WEIGHT_LABELS[weighting]}"
            schema_fingerprint = module_report.schema_fingerprint
            corpus_work_count = module_report.work_count
            table_filename = MODULE_METRICS_FILENAME

        print("\nWhat would you like to find?")
        mode = prompt_choice(
            "",
            [
                ("single", "Highest/lowest poems for one selected metric"),
                ("pair", "HIGH/LOW combination across two selected metrics"),
                ("general", "Broad anomaly scan with short explanations"),
            ],
            default=1,
        )

        run_folder = create_run_folder(output_root, source, table_key)
        base_metadata: dict[str, object] = {
            "tool": "VerseVAD Anomaly Finder",
            "tool_version": __version__,
            "reader_version": getattr(versevad_reader, "__version__", "unknown"),
            "created_at_local": datetime.now().isoformat(timespec="seconds"),
            "source_path": str(source),
            "source_filename": source.name,
            "source_sha256": source_sha256(source),
            "evidence_table": table_filename,
            "evidence_table_mode": table_key,
            "schema_fingerprint": schema_fingerprint,
            "corpus_work_count": corpus_work_count,
            "profile": profile_label,
            "analysis_view": analysis_view,
            "weighting": weighting,
            "coverage_threshold": coverage_threshold,
            "percentile_method": "average rank: (rank - 0.5) / N * 100",
            "z_score_method": "corpus-standardized with population SD (ddof=0)",
            "interpretive_caution": (
                "Anomaly means corpus-relative unusualness only; it is not a significance test and "
                "does not imply error, pathology, artistic value, or causal importance."
            ),
        }

        if mode == "single":
            pool: Sequence[object] = choices if table_key == "vad" else module_choices
            selected = select_metric_interactively(pool, "metric")
            if selected is None:
                print("No metric selected. Analysis cancelled.")
                return 0
            direction = prompt_choice(
                "\nWhich tail do you want?",
                [("high", "Highest"), ("low", "Lowest"), ("both", "Both highest and lowest")],
                default=3,
            )
            result_count = prompt_result_count()
            print(f"\nRunning single-metric anomaly search: {getattr(selected, 'label')}...")
            if table_key == "vad":
                assert isinstance(selected, LexicalMetricChoice)
                results, all_ranked = analyze_lexical_single(reader, selected, direction, result_count, coverage_threshold)
                metric_payload = asdict(selected)
                concept = selected.concept_label
            else:
                assert isinstance(selected, ModuleMetricChoice)
                results, all_ranked = analyze_module_single(module_reader, selected, direction, result_count)
                metric_payload = asdict(selected)
                concept = selected.concept_label
            print_single_results(results, concept, show_coverage=(table_key == "vad"))
            metadata = {
                **base_metadata,
                "mode": "single_metric_extremes",
                "metric": metric_payload,
                "direction": direction,
                "result_count_per_tail": result_count,
                "eligible_poems": len(all_ranked),
            }
            spec = {
                "mode": "single_metric_extremes",
                "evidence_table": table_filename,
                "profile": {"analysis_view": analysis_view, "weighting": weighting},
                "coverage_threshold": coverage_threshold,
                "metric": metric_payload,
                "direction": direction,
                "result_count_per_tail": result_count,
            }
            export_single_mode(run_folder, results, all_ranked, metadata)

        elif mode == "pair":
            pool = choices if table_key == "vad" else module_choices
            a = select_metric_interactively(pool, "Metric A")
            if a is None:
                print("No metric selected. Analysis cancelled.")
                return 0
            b = select_metric_interactively(pool, "Metric B")
            if b is None:
                print("No second metric selected. Analysis cancelled.")
                return 0
            if getattr(a, "identity_key") == getattr(b, "identity_key"):
                raise ValueError("Metric A and Metric B must be different metrics.")
            combo = prompt_choice(
                "\nChoose directional combination:",
                [
                    ("high_high", f"HIGH {getattr(a, 'concept_label')} + HIGH {getattr(b, 'concept_label')}"),
                    ("high_low", f"HIGH {getattr(a, 'concept_label')} + LOW {getattr(b, 'concept_label')}"),
                    ("low_high", f"LOW {getattr(a, 'concept_label')} + HIGH {getattr(b, 'concept_label')}"),
                    ("low_low", f"LOW {getattr(a, 'concept_label')} + LOW {getattr(b, 'concept_label')}"),
                ],
                default=1,
            )
            direction_a, direction_b = combo.split("_")
            extremity_threshold = prompt_pair_extremity()
            result_count = prompt_result_count()
            print(
                f"\nRunning directional search: {direction_a.upper()} {getattr(a, 'concept_label')} + "
                f"{direction_b.upper()} {getattr(b, 'concept_label')}..."
            )
            if table_key == "vad":
                assert isinstance(a, LexicalMetricChoice) and isinstance(b, LexicalMetricChoice)
                matches, ranked = analyze_lexical_pair(
                    reader, a, b, direction_a, direction_b, extremity_threshold, result_count, coverage_threshold
                )
            else:
                assert isinstance(a, ModuleMetricChoice) and isinstance(b, ModuleMetricChoice)
                matches, ranked = analyze_module_pair(
                    module_reader, a, b, direction_a, direction_b, extremity_threshold, result_count
                )
            print_pair_results(
                matches, ranked, getattr(a, "concept_label"), getattr(b, "concept_label"),
                direction_a, direction_b, extremity_threshold, result_count,
            )
            metadata = {
                **base_metadata,
                "mode": "two_metric_directional_combination",
                "metric_a": asdict(a),
                "metric_b": asdict(b),
                "direction_a": direction_a,
                "direction_b": direction_b,
                "directional_extremity_threshold": extremity_threshold,
                "result_count": result_count,
                "eligible_paired_poems": len(ranked),
                "strict_matches": int(ranked["meets_threshold"].sum()),
                "joint_directional_score": (
                    "geometric mean of the two directional percentile-extremity scores; threshold "
                    "matching still requires BOTH directions individually"
                ),
            }
            spec = {
                "mode": "two_metric_directional_combination",
                "evidence_table": table_filename,
                "profile": {"analysis_view": analysis_view, "weighting": weighting},
                "coverage_threshold": coverage_threshold,
                "metric_a": asdict(a),
                "metric_b": asdict(b),
                "direction_a": direction_a,
                "direction_b": direction_b,
                "directional_extremity_threshold": extremity_threshold,
                "result_count": result_count,
            }
            export_pair_mode(run_folder, matches, ranked, metadata)

        else:
            result_count = prompt_result_count(default=30)
            rare_categories: Optional[pd.DataFrame] = None
            if table_key == "vad":
                scan_count = sum(is_lexical_general_metric(c.metric) for c in choices)
                print(
                    "\nBroad lexical scan design:\n"
                    f"  Candidate mean/rate/dispersion metric identities: {scan_count}\n"
                    "  Includes SD/deviation measures when semantically meaningful.\n"
                    "  Excludes coverage, cumulative/load, and generic length-sensitive totals.\n"
                    "  Parallel resources for the same concept are collapsed by median percentile."
                )
                if not prompt_yes_no("Run the broad anomaly scan?", default=True):
                    print("Analysis cancelled.")
                    return 0
                print("\nRunning broad lexical anomaly scan...")
                results, scores, evidence, retained = analyze_lexical_general(
                    reader, choices, result_count, coverage_threshold
                )
                metadata = {
                    **base_metadata,
                    "mode": "broad_lexical_anomaly_scan",
                    "candidate_metric_identities": scan_count,
                    "retained_metric_identities": len(retained),
                    "result_count": result_count,
                    "concept_collapse_rule": "median percentile across parallel resources sharing the same concept label",
                    "overall_score": "mean percentile-extremeness across eligible conceptual dimensions",
                    "retained_metrics": [asdict(c) for c in retained],
                }
                spec = {
                    "mode": "broad_lexical_anomaly_scan",
                    "evidence_table": table_filename,
                    "profile": {"analysis_view": analysis_view, "weighting": weighting},
                    "coverage_threshold": coverage_threshold,
                    "result_count": result_count,
                }
            else:
                include_size = prompt_yes_no(
                    "Include raw length/size-sensitive counts in the broad scan (word count, syllable count, rhyme-pair counts, etc.)?",
                    default=False,
                )
                include_categories = prompt_yes_no(
                    "Also surface rare low-cardinality categorical form/meter labels?",
                    default=True,
                )
                numeric_scan_count = sum(module_general_numeric_eligible(c, include_size) for c in module_choices)
                print(
                    "\nBroad module scan design:\n"
                    f"  Candidate numeric document-level metrics: {numeric_scan_count}\n"
                    "  Includes SD/IQR/variability, rhyme density, meter fit, readability, lexical diversity, form, and sentiment.\n"
                    "  Excludes technical VerseMap coordinates, evidence/coverage diagnostics, and cumulative loads from the automatic score.\n"
                    "  Broad profile scores are balanced by module so modules with many exported metrics do not dominate."
                )
                if not prompt_yes_no("Run the broad anomaly scan?", default=True):
                    print("Analysis cancelled.")
                    return 0
                print("\nRunning broad multi-module anomaly scan. This may take a little while...")
                results, scores, evidence, rare_categories, retained_numeric, retained_categories = analyze_module_general(
                    module_reader,
                    module_choices,
                    analysis_view,
                    weighting,
                    result_count,
                    include_size,
                    include_categories,
                )
                metadata = {
                    **base_metadata,
                    "mode": "broad_module_anomaly_scan",
                    "candidate_numeric_metric_identities": numeric_scan_count,
                    "retained_numeric_metric_identities": len(retained_numeric),
                    "categorical_anomaly_scan": include_categories,
                    "categorical_metric_identities_considered": len(retained_categories),
                    "include_size_sensitive_counts": include_size,
                    "result_count": result_count,
                    "numeric_metric_extremeness": "2 * abs(percentile - 50)",
                    "broad_profile_score": "mean metric extremeness within module, then mean across modules",
                    "top_numeric_score": f"mean of the {GENERAL_TOP_COMPONENTS} strongest numeric concept extremeness values",
                    "rare_category_rule": f"category share <= {MODULE_RARE_CATEGORY_MAX_SHARE:.2f} and <= {MODULE_RARE_CATEGORY_MAX_LEVELS} category levels",
                    "overall_anomaly_score": "module-balanced broad numeric profile score; categorical rarity is reported separately rather than folded into the score",
                    "rare_category_output": "general_rare_categories.csv when qualifying categories are found",
                    "retained_numeric_metrics": [asdict(c) for c in retained_numeric],
                }
                spec = {
                    "mode": "broad_module_anomaly_scan",
                    "evidence_table": table_filename,
                    "preferred_profile": {"analysis_view": analysis_view, "weighting": weighting},
                    "include_size_sensitive_counts": include_size,
                    "include_rare_categories": include_categories,
                    "result_count": result_count,
                }
            print_general_results(results, table_key)
            if table_key == "module" and rare_categories is not None and not rare_categories.empty:
                print("\nRare categorical patterns")
                print("-------------------------")
                for _, row in rare_categories.head(10).iterrows():
                    print(
                        f"{int(row['rarity_rank']):>2}. {row['title']} | {row['metric_label']} = {row['value']} "
                        f"| corpus share={float(row['category_share']) * 100:.1f}%"
                    )
            export_general_mode(run_folder, results, scores, evidence, metadata, rare_categories)

        with (run_folder / "analysis_spec.json").open("w", encoding="utf-8") as handle:
            json.dump(json_ready(spec), handle, indent=2, ensure_ascii=False, sort_keys=True)
        with (run_folder / "analysis_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(json_ready(metadata), handle, indent=2, ensure_ascii=False, sort_keys=True)

        print("\nExports written")
        print("===============")
        print(run_folder)
        print("\nFiles include:")
        print("  anomaly_results.csv")
        print("  anomaly_analysis.xlsx")
        print("  analysis_spec.json")
        print("  analysis_metadata.json")
        if mode == "single":
            print("  single_metric_all_ranked.csv")
        elif mode == "pair":
            print("  pair_directional_all_ranked.csv")
        else:
            print("  general_anomaly_all_scores.csv")
            print("  general_anomaly_evidence.csv")
            if table_key == "module" and rare_categories is not None and not rare_categories.empty:
                print("  general_rare_categories.csv")
        return 0

    except (VerseVADReaderError, FileNotFoundError, ValueError, zipfile.BadZipFile) as exc:
        print(f"\nVERSEVAD ANOMALY ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nAnalysis cancelled by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
