#!/usr/bin/env python3
"""compare.py

Interactive multi-corpus comparison for VerseVAD Complete Audit exports.

Designed to live beside ``versevad_reader.py`` inside::

    versevad_stats/
        scripts/
            versevad_reader.py
            correlation.py
            anomaly.py
            sensitivity.py
            robustness.py
            compare.py
        source/
            <2-5 VerseVAD Complete Audit ZIPs or compatible CSVs>
        exports/

Run with::

    python scripts/compare.py

Research questions
------------------
The program is organized around three increasingly comparative questions:

1. What does each corpus look like?
   Report work-level descriptive distributions for each selected metric under
   exactly the same VerseVAD resource, lexical scope, and weighting choices.

2. How far apart are these corpora?
   Report every unique corpus pair using signed raw mean/median differences,
   bootstrap confidence intervals for the mean difference, and exact sample /
   coverage information.

3. Which measured qualities distinguish the corpora most strongly?
   Report Cliff's delta as a scale-independent effect size and rank selected
   measurements by the absolute magnitude of that effect.

Comparison discipline
---------------------
Methodological choices are universal within a run. A comparison is created only
when the SAME metric, SAME lexicon/resource, SAME lexical scope, SAME weighting,
and compatible exported scale exist in every selected corpus. The script never
silently substitutes a different VAD lexicon or profile for one corpus.

If multiple compatible VAD resources are selected, each resource is analyzed as
its own parallel comparison. VerseVAD's exported normalized 0-1 VAD means are
used directly; no second normalization is performed.

The corpus-level descriptive mean is always an equal-work mean of poem/work-level
VerseVAD values. Long poems therefore do not receive extra corpus weight merely
because they contain more tokens.

No formal hypothesis testing is performed. There are no Mann-Whitney tests,
p-values, or multiple-testing corrections in this script. Bootstrap confidence
intervals are estimation tools here, not significance labels.

Selection syntax
----------------
Multiple-selection prompts accept::

    A          all choices (when allowed)
    1          choice 1
    1,2        choices 1 and 2
    1,3,5      choices 1, 3, and 5
    5-6        choices 5 through 6
    1,3,5-6    mixed individual choices and ranges

Principal outputs
-----------------
Each run writes to ``exports/compare_corpus/<run>/``::

    corpus_results.xlsx
    pairwise_differences.xlsx
    corpus_metric_results.csv
    across_corpus_summary.csv
    coverage_summary.csv
    pairwise_differences.csv
    largest_differences.csv
    metric_distinguishing_summary.csv
    pairwise_effect_sizes.csv
    work_values_long.csv
    analysis_spec.json
    analysis_metadata.json

Dependencies
------------
Python 3.10+, numpy, pandas, and ``versevad_reader.py``.
The XLSX writer uses only Python's standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Optional, Sequence
from xml.sax.saxutils import escape as xml_escape

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "compare.py requires numpy and pandas.\n"
        "Install them with:\n"
        "  python -m pip install numpy pandas"
    ) from exc

try:
    import versevad_reader
    from versevad_reader import VerseVADCorpusReader
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Could not import versevad_reader.py. Put compare.py beside "
        "versevad_reader.py inside the scripts folder."
    ) from exc

from versevad_tools.cli import parse_coverage_threshold, parse_index_selection
from versevad_tools.core import (
    configure_console_encoding,
    file_sha256,
    json_ready,
    pretty_words,
    project_root as shared_project_root,
    slugify as shared_slugify,
)
from versevad_tools.metrics import (
    RESOURCE_SHORT_NAMES,
    VAD_RESOURCE_IDS,
    VIEW_LABELS,
    WEIGHTING_LABELS,
    friendly_metric_label,
    profile_label,
    resource_label,
    statistic_suffix,
)
from versevad_tools.sources import discover_corpus_metric_sources


__version__ = "0.1.0"
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_CONFIDENCE = 0.95
DEFAULT_RANDOM_SEED = 12_345

IDENTITY_COLUMNS = (
    "lexicon_id",
    "metric",
    "dimension",
    "category",
    "analysis_view",
    "weighting",
)
LOAD_COLUMNS = (
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
    "analysis_view",
)

# Conventional descriptive Cliff's-delta magnitude cutoffs. These are aids for
# reading the effect-size column, not hard boundaries in literary interpretation.
CLIFF_THRESHOLDS = (
    (0.147, "negligible"),
    (0.330, "small"),
    (0.474, "medium"),
    (math.inf, "large"),
)


@dataclass(frozen=True)
class MetricConcept:
    label: str
    metric: str
    dimension: str
    category: str
    common_resource_ids: tuple[str, ...]
    common_resource_labels: tuple[str, ...]

    @property
    def key(self) -> str:
        return "|".join((self.metric, self.dimension, self.category))

    @property
    def search_blob(self) -> str:
        bits = [
            self.label,
            self.metric,
            self.dimension,
            self.category,
            *self.common_resource_ids,
            *self.common_resource_labels,
        ]
        return " ".join(str(x) for x in bits if str(x).strip()).casefold()


@dataclass(frozen=True)
class ComparisonVariant:
    metric_key: str
    metric_label: str
    lexicon_id: str
    lexicon: str
    metric: str
    dimension: str
    category: str
    analysis_view: str
    weighting: str
    scale: str

    @property
    def resource_label(self) -> str:
        return resource_label(self.lexicon_id, self.lexicon)

    @property
    def profile_label(self) -> str:
        return profile_label(self.analysis_view, self.weighting)

    @property
    def exact_key(self) -> tuple[str, str, str, str, str, str]:
        return (
            self.lexicon_id,
            self.metric,
            self.dimension,
            self.category,
            self.analysis_view,
            self.weighting,
        )

    @property
    def variant_id(self) -> str:
        return "|".join(self.exact_key)

    @property
    def label(self) -> str:
        return f"{self.metric_label} · {self.resource_label} · {self.profile_label}"


@dataclass
class CorpusInput:
    path: Path
    label: str
    reader: VerseVADCorpusReader
    report: object
    catalog: pd.DataFrame


@dataclass(frozen=True)
class StyledCell:
    value: object
    style_id: int


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def project_root() -> Path:
    return shared_project_root(__file__)


def slugify(text: str) -> str:
    return shared_slugify(text, fallback="comparison", max_length=72)


def default_corpus_label(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"(?i)[_\- ]*versevad[_\- ]*complete[_\- ]*audit.*$", "", stem)
    stem = re.sub(r"(?i)[_\- ]*complete[_\- ]*audit.*$", "", stem)
    stem = re.sub(r"(?i)[_\- ]*versevad.*$", "", stem)
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem or path.stem


def signed_display(value: float, decimals: int = 6) -> str:
    if not math.isfinite(value):
        return ""
    if abs(value) < 0.5 * (10 ** -decimals):
        value = 0.0
    return f"{value:+.{decimals}f}"


def cliff_magnitude(delta: float) -> str:
    if not math.isfinite(delta):
        return ""
    magnitude = abs(delta)
    for threshold, label in CLIFF_THRESHOLDS:
        if magnitude < threshold:
            return label
    return "large"


def cliff_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Return Cliff's delta for x versus y without hypothesis testing.

    +1 means every x observation exceeds every y observation.
    -1 means every x observation is below every y observation.
    0 means neither sample tends to dominate the other.
    Ties contribute zero to the numerator.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return math.nan
    ys = np.sort(y)
    less = np.searchsorted(ys, x, side="left")
    greater = y.size - np.searchsorted(ys, x, side="right")
    return float((less.sum() - greater.sum()) / (x.size * y.size))


def bootstrap_mean_difference_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_resamples: int,
    confidence: float,
    rng: np.random.Generator,
    batch_size: int = 1000,
) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0 or n_resamples <= 0:
        return math.nan, math.nan

    estimates = np.empty(n_resamples, dtype=float)
    written = 0
    while written < n_resamples:
        b = min(batch_size, n_resamples - written)
        x_idx = rng.integers(0, x.size, size=(b, x.size))
        y_idx = rng.integers(0, y.size, size=(b, y.size))
        estimates[written : written + b] = x[x_idx].mean(axis=1) - y[y_idx].mean(axis=1)
        written += b

    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return float(low), float(high)


# ---------------------------------------------------------------------------
# Source discovery and validation
# ---------------------------------------------------------------------------


def discover_sources(directory: Path) -> list[Path]:
    return discover_corpus_metric_sources(directory, versevad_reader.METRICS_FILENAME)


def choose_sources(paths: Sequence[Path]) -> list[Path]:
    if len(paths) < 2:
        raise SystemExit(
            "compare.py needs at least two VerseVAD corpus sources in the source folder.\n"
            "Add Complete Audit ZIPs (or compatible standalone corpus_vad_metrics.csv files) and try again."
        )
    print("\n# Select corpora")
    for idx, path in enumerate(paths, start=1):
        print(f"[{idx}] {path.name}")
    print("\nSelect 2-5 corpora using 1,2, 1-3, 1,3,5, etc.")
    if len(paths) <= 5:
        print("A selects all available corpora.")
    while True:
        raw = input("Selection: ").strip()
        try:
            indexes = parse_index_selection(
                raw,
                len(paths),
                min_count=2,
                max_count=5,
                allow_all=len(paths) <= 5,
            )
            return [paths[i] for i in indexes]
        except ValueError as exc:
            print(str(exc))


def maybe_customize_labels(paths: Sequence[Path]) -> list[str]:
    defaults = [default_corpus_label(path) for path in paths]
    print("\nCorpus labels used in outputs:")
    for idx, label in enumerate(defaults, start=1):
        print(f"[{idx}] {label}")
    raw = input("Customize these labels? [y/N]: ").strip().casefold()
    if raw not in {"y", "yes"}:
        labels = defaults
    else:
        labels = []
        for idx, default in enumerate(defaults, start=1):
            value = input(f"Label for corpus {idx} [{default}]: ").strip()
            labels.append(value or default)

    seen: dict[str, int] = {}
    unique: list[str] = []
    for label in labels:
        key = label.casefold()
        seen[key] = seen.get(key, 0) + 1
        if seen[key] == 1:
            unique.append(label)
        else:
            unique.append(f"{label} ({seen[key]})")
    return unique


def validate_inputs(paths: Sequence[Path], labels: Sequence[str]) -> list[CorpusInput]:
    out: list[CorpusInput] = []
    print("\n# Validate selected corpora")
    for path, label in zip(paths, labels):
        print(f"\n{label}")
        reader = VerseVADCorpusReader(path)
        report = reader.validate(strict_schema=False)
        if not report.valid:
            raise SystemExit(f"Validation failed for {path.name}. Refusing to compare invalid corpus data.")
        catalog = reader.catalog().fillna("")
        print(
            f"  {report.work_count:,} works · {report.lexicon_count} lexical resources · "
            f"{report.metric_identity_count} metric identities"
        )
        out.append(CorpusInput(path=path, label=label, reader=reader, report=report, catalog=catalog))
    return out


# ---------------------------------------------------------------------------
# Common metric/resource/profile selection
# ---------------------------------------------------------------------------


def concept_rows(catalog: pd.DataFrame) -> dict[str, pd.DataFrame]:
    work = catalog.copy().fillna("")
    result: dict[str, pd.DataFrame] = {}
    for (metric, dimension, category), group in work.groupby(
        ["metric", "dimension", "category"], sort=False, dropna=False
    ):
        key = "|".join((str(metric), str(dimension), str(category)))
        result[key] = group.copy()
    return result


def build_common_metric_concepts(corpora: Sequence[CorpusInput]) -> list[MetricConcept]:
    maps = [concept_rows(c.catalog) for c in corpora]
    common_keys = set(maps[0])
    for mapping in maps[1:]:
        common_keys &= set(mapping)

    concepts: list[MetricConcept] = []
    for key in common_keys:
        groups = [mapping[key] for mapping in maps]
        resource_sets = [set(g["lexicon_id"].astype(str)) for g in groups]
        common_resources = set.intersection(*resource_sets) if resource_sets else set()
        if not common_resources:
            continue
        representative = groups[0].iloc[0]
        resource_pairs: list[tuple[str, str]] = []
        for rid in sorted(common_resources):
            rows = groups[0].loc[groups[0]["lexicon_id"].astype(str) == rid]
            lexicon = str(rows.iloc[0]["lexicon"]) if not rows.empty else rid
            resource_pairs.append((rid, resource_label(rid, lexicon)))
        metric, dimension, category = key.split("|", 2)
        concepts.append(
            MetricConcept(
                label=friendly_metric_label(representative),
                metric=metric,
                dimension=dimension,
                category=category,
                common_resource_ids=tuple(rid for rid, _ in resource_pairs),
                common_resource_labels=tuple(label for _, label in resource_pairs),
            )
        )
    concepts.sort(key=lambda c: c.label.casefold())
    return concepts


def search_metric_concepts(concepts: Sequence[MetricConcept], query: str) -> list[MetricConcept]:
    terms = [term.casefold() for term in re.findall(r"\S+", query.strip())]
    if not terms:
        return list(concepts)
    scored: list[tuple[int, str, MetricConcept]] = []
    for concept in concepts:
        blob = concept.search_blob
        if not all(term in blob for term in terms):
            continue
        label = concept.label.casefold()
        score = 0
        if query.strip().casefold() == label:
            score -= 100
        if all(term in label for term in terms):
            score -= 20
        if any(label.startswith(term) for term in terms):
            score -= 5
        scored.append((score, label, concept))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in scored]


def choose_metrics(concepts: Sequence[MetricConcept]) -> list[MetricConcept]:
    if not concepts:
        raise SystemExit("The selected corpora have no directly compatible VerseVAD metric concepts.")
    selected: list[MetricConcept] = []
    selected_keys: set[str] = set()
    print("\n# Select metrics")
    print(
        "Only metrics with at least one SAME compatible resource across every selected corpus are offered.\n"
        "Search terms can include concreteness, visual, interoceptive, frequency, aoa, valence, arousal, etc."
    )
    print("Type Q when finished.\n")

    while True:
        if selected:
            print("Currently selected:")
            for idx, concept in enumerate(selected, start=1):
                print(f"  {idx}. {concept.label}")
            print()
        query = input("Search metric (Q to finish): ").strip()
        if query.casefold() == "q":
            if selected:
                return selected
            print("Select at least one metric before continuing.")
            continue
        matches = search_metric_concepts(concepts, query)
        if not matches:
            print("No compatible matching metrics. Try broader search terms.\n")
            continue
        shown = matches[:25]
        print(f"\nMatches ({len(matches)} total):")
        for idx, concept in enumerate(shown, start=1):
            resources = ", ".join(concept.common_resource_labels)
            print(f"[{idx}] {concept.label}")
            print(f"    compatible resources across ALL selected corpora: {resources}")
        if len(matches) > len(shown):
            print(f"    ... {len(matches)-len(shown)} more; refine the search to narrow them.")
        raw = input("Select number, R to refine, or Q to finish: ").strip()
        if raw.casefold() == "r":
            print()
            continue
        if raw.casefold() == "q":
            if selected:
                return selected
            print("Select at least one metric before continuing.\n")
            continue
        try:
            idx = int(raw)
            if not 1 <= idx <= len(shown):
                raise ValueError
        except ValueError:
            print("Enter one of the displayed numbers, R, or Q.\n")
            continue
        concept = shown[idx - 1]
        if concept.key in selected_keys:
            print(f"Already selected: {concept.label}\n")
            continue
        selected.append(concept)
        selected_keys.add(concept.key)
        print(f"Added: {concept.label}\n")


def resources_for_concept_across_corpora(corpora: Sequence[CorpusInput], concept: MetricConcept) -> list[tuple[str, str]]:
    resource_sets: list[set[str]] = []
    labels: dict[str, str] = {}
    for corpus in corpora:
        cat = corpus.catalog.fillna("")
        rows = cat.loc[
            (cat["metric"].astype(str) == concept.metric)
            & (cat["dimension"].astype(str) == concept.dimension)
            & (cat["category"].astype(str) == concept.category)
        ]
        ids = set(rows["lexicon_id"].astype(str))
        resource_sets.append(ids)
        for row in rows[["lexicon_id", "lexicon"]].drop_duplicates().itertuples(index=False):
            labels.setdefault(str(row.lexicon_id), resource_label(str(row.lexicon_id), str(row.lexicon)))
    common = set.intersection(*resource_sets) if resource_sets else set()
    return sorted(((rid, labels.get(rid, rid)) for rid in common), key=lambda x: x[1].casefold())


def choose_resources(
    corpora: Sequence[CorpusInput],
    concepts: Sequence[MetricConcept],
) -> dict[str, list[str]]:
    """Choose comparison resources, applying VAD lexicon choices universally.

    A VAD resource selection is made once and then applied to every selected VAD
    metric (Valence, Arousal, Dominance, VAD dispersion/load metrics, etc.) for
    which that exact resource is jointly available in every selected corpus.
    Non-VAD single-resource metrics remain automatic.
    """
    result: dict[str, list[str]] = {}
    print("\n# Select lexicon/resources")
    print(
        "Resource choices are universal across corpora. A selected resource must exist for the SAME metric in every corpus.\n"
        "For VAD, one lexicon selection is applied to ALL selected VAD metrics. Single-resource non-VAD metrics are automatic."
    )

    resources_by_concept = {
        concept.key: resources_for_concept_across_corpora(corpora, concept)
        for concept in concepts
    }

    # Group VAD concepts so the lexicon choice is made once for the whole VAD family.
    vad_concepts: list[MetricConcept] = []
    for concept in concepts:
        ids = {rid for rid, _ in resources_by_concept[concept.key]}
        if ids and ids.issubset(VAD_RESOURCE_IDS):
            vad_concepts.append(concept)

    if vad_concepts:
        common_vad_ids: Optional[set[str]] = None
        label_lookup: dict[str, str] = {}
        for concept in vad_concepts:
            resources = resources_by_concept[concept.key]
            ids = {rid for rid, _ in resources}
            common_vad_ids = ids if common_vad_ids is None else common_vad_ids & ids
            for rid, label in resources:
                label_lookup.setdefault(rid, label)
        if not common_vad_ids:
            names = ", ".join(c.label for c in vad_concepts)
            raise RuntimeError(
                "The selected VAD metrics do not share a single compatible VAD resource across every corpus: " + names
            )
        vad_resources = sorted(
            ((rid, label_lookup.get(rid, rid)) for rid in common_vad_ids),
            key=lambda x: x[1].casefold(),
        )
        print("\nVAD lexicon selection (applies to ALL selected VAD metrics):")
        if len(vad_resources) == 1:
            selected_vad_ids = [vad_resources[0][0]]
            print(f"{vad_resources[0][1]} [automatic]")
        else:
            for idx, (_, label) in enumerate(vad_resources, start=1):
                print(f"[{idx}] {label}")
            print("Select one or more using A, 1,2, 1,3, or ranges such as 1-3.")
            while True:
                raw = input("Selection: ").strip()
                try:
                    indexes = parse_index_selection(raw, len(vad_resources))
                    selected_vad_ids = [vad_resources[i][0] for i in indexes]
                    break
                except ValueError as exc:
                    print(str(exc))
        for concept in vad_concepts:
            result[concept.key] = list(selected_vad_ids)
            print(
                f"  {concept.label}: "
                + ", ".join(label_lookup.get(rid, rid) for rid in selected_vad_ids)
            )

    # Handle non-VAD concepts. In normal VerseVAD use these are usually
    # single-resource metrics such as Brysbaert, Lancaster, SUBTLEX, and Kuperman.
    for concept in concepts:
        if concept in vad_concepts:
            continue
        resources = resources_by_concept[concept.key]
        if not resources:
            raise RuntimeError(f"No common resource remains for {concept.label}.")
        if len(resources) == 1:
            result[concept.key] = [resources[0][0]]
            print(f"\n{concept.label}: {resources[0][1]} [automatic]")
            continue
        print(f"\n{concept.label} is available from multiple SAME resources across all selected corpora:")
        for idx, (_, label) in enumerate(resources, start=1):
            print(f"[{idx}] {label}")
        print("Select one or more using A, 1,2, 1,3, or ranges such as 1-3.")
        while True:
            raw = input("Selection: ").strip()
            try:
                indexes = parse_index_selection(raw, len(resources))
                result[concept.key] = [resources[i][0] for i in indexes]
                break
            except ValueError as exc:
                print(str(exc))
    return result


def common_profiles(
    corpora: Sequence[CorpusInput],
    concepts: Sequence[MetricConcept],
    resources: dict[str, list[str]],
) -> list[tuple[str, str]]:
    requirement_profile_sets: list[set[tuple[str, str]]] = []
    for concept in concepts:
        for rid in resources[concept.key]:
            for corpus in corpora:
                cat = corpus.catalog.fillna("")
                rows = cat.loc[
                    (cat["metric"].astype(str) == concept.metric)
                    & (cat["dimension"].astype(str) == concept.dimension)
                    & (cat["category"].astype(str) == concept.category)
                    & (cat["lexicon_id"].astype(str) == rid)
                ]
                profiles = {
                    (str(row.analysis_view), str(row.weighting))
                    for row in rows.itertuples()
                }
                requirement_profile_sets.append(profiles)
    if not requirement_profile_sets:
        return []
    common = set.intersection(*requirement_profile_sets)
    view_order = {view: idx for idx, view in enumerate(VIEW_LABELS)}
    weight_order = {w: idx for idx, w in enumerate(WEIGHTING_LABELS)}
    return sorted(common, key=lambda p: (view_order.get(p[0], 99), weight_order.get(p[1], 99)))


def choose_profiles(profiles: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    if not profiles:
        raise SystemExit(
            "No single scope/weight profile is available for every selected metric/resource across every selected corpus."
        )
    print("\n# Choose lexical scope / weighting profiles")
    print("Only profiles available for EVERY selected metric/resource in EVERY corpus are shown.")
    for idx, profile in enumerate(profiles, start=1):
        print(f"[{idx}] {profile_label(*profile)}")
    print("\nSelect using A, 1,3,5, 5-6, or combinations such as 1,3,5-6.")
    print("Press Enter to include all compatible profiles.")
    while True:
        raw = input("Selection: ").strip()
        if not raw:
            return list(profiles)
        try:
            indexes = parse_index_selection(raw, len(profiles))
            return [profiles[i] for i in indexes]
        except ValueError as exc:
            print(str(exc))


def _row_for_exact_variant(
    catalog: pd.DataFrame,
    concept: MetricConcept,
    rid: str,
    profile: tuple[str, str],
) -> pd.DataFrame:
    cat = catalog.fillna("")
    return cat.loc[
        (cat["metric"].astype(str) == concept.metric)
        & (cat["dimension"].astype(str) == concept.dimension)
        & (cat["category"].astype(str) == concept.category)
        & (cat["lexicon_id"].astype(str) == rid)
        & (cat["analysis_view"].astype(str) == profile[0])
        & (cat["weighting"].astype(str) == profile[1])
    ]


def build_comparison_variants(
    corpora: Sequence[CorpusInput],
    concepts: Sequence[MetricConcept],
    resources: dict[str, list[str]],
    profiles: Sequence[tuple[str, str]],
) -> list[ComparisonVariant]:
    variants: list[ComparisonVariant] = []
    skipped: list[str] = []
    for concept in concepts:
        for rid in resources[concept.key]:
            for profile in profiles:
                exact_rows = [_row_for_exact_variant(c.catalog, concept, rid, profile) for c in corpora]
                if any(rows.empty for rows in exact_rows):
                    skipped.append(f"{concept.label} · {rid} · {profile_label(*profile)} [missing exact variant]")
                    continue
                scale_sets = [set(rows["scale"].astype(str).str.strip()) - {""} for rows in exact_rows]
                all_scales = set().union(*scale_sets)
                compatible = len(all_scales) <= 1
                if not compatible:
                    # Defensive allowance for cosmetically different labels that all explicitly
                    # identify a normalized 0-1 VAD scale.
                    if rid in VAD_RESOURCE_IDS:
                        compatible = all(
                            all("normal" in s.casefold() and "0" in s and "1" in s for s in scales)
                            for scales in scale_sets if scales
                        )
                if not compatible:
                    skipped.append(
                        f"{concept.label} · {rid} · {profile_label(*profile)} [incompatible scales: {sorted(all_scales)}]"
                    )
                    continue
                first = exact_rows[0].iloc[0]
                scale = str(first.get("scale", "") or "")
                variants.append(
                    ComparisonVariant(
                        metric_key=concept.key,
                        metric_label=concept.label,
                        lexicon_id=rid,
                        lexicon=str(first.get("lexicon", "") or ""),
                        metric=concept.metric,
                        dimension=concept.dimension,
                        category=concept.category,
                        analysis_view=profile[0],
                        weighting=profile[1],
                        scale=scale,
                    )
                )
    if skipped:
        print("\nCompatibility notes:")
        for note in skipped[:20]:
            print(f"  - {note}")
        if len(skipped) > 20:
            print(f"  ... {len(skipped)-20} additional incompatible variants omitted.")
    if not variants:
        raise SystemExit("No exact like-for-like comparison variants remain after compatibility checks.")
    return variants


# ---------------------------------------------------------------------------
# Efficient loading
# ---------------------------------------------------------------------------


def _identity_string(frame: pd.DataFrame) -> pd.Series:
    cols = [frame[col].fillna("").astype(str) for col in IDENTITY_COLUMNS]
    out = cols[0]
    for col in cols[1:]:
        out = out + "\x1f" + col
    return out


def _variant_identity_string(variant: ComparisonVariant) -> str:
    return "\x1f".join(variant.exact_key)


def load_selected_variants(
    corpus: CorpusInput,
    variants: Sequence[ComparisonVariant],
) -> pd.DataFrame:
    identity_to_variant = {_variant_identity_string(v): v for v in variants}
    wanted = set(identity_to_variant)
    pieces: list[pd.DataFrame] = []

    for chunk in corpus.reader._iter_chunks(usecols=LOAD_COLUMNS):  # intentional streaming use
        chunk = chunk.copy()
        for col in ("dimension", "category"):
            chunk[col] = chunk[col].fillna("").astype(str)
        ids = _identity_string(chunk)
        mask = ids.isin(wanted)
        if not mask.any():
            continue
        selected = chunk.loc[mask].copy()
        selected["_exact_identity"] = ids.loc[mask].values
        selected["variant_id"] = selected["_exact_identity"].map(
            lambda key: identity_to_variant[str(key)].variant_id
        )
        selected["metric_key"] = selected["_exact_identity"].map(
            lambda key: identity_to_variant[str(key)].metric_key
        )
        selected["metric_label"] = selected["_exact_identity"].map(
            lambda key: identity_to_variant[str(key)].metric_label
        )
        selected["resource_label"] = selected["_exact_identity"].map(
            lambda key: identity_to_variant[str(key)].resource_label
        )
        selected["profile_label"] = selected["_exact_identity"].map(
            lambda key: identity_to_variant[str(key)].profile_label
        )
        selected = selected.drop(columns=["_exact_identity"])
        pieces.append(selected)

    if not pieces:
        raise RuntimeError(f"No selected rows were found for corpus {corpus.label}.")

    data = pd.concat(pieces, ignore_index=True)
    numeric_cols = ["value", "observations", "matched_tokens", "lexical_tokens", "coverage"]
    for col in numeric_cols:
        raw = data[col].astype(str).str.strip()
        converted = pd.to_numeric(raw.replace("", np.nan), errors="coerce")
        invalid = raw.ne("") & converted.isna()
        if invalid.any():
            sample = data.loc[invalid, ["title", "metric_label", "resource_label", "profile_label", col]].head(5)
            raise RuntimeError(
                f"Selected rows in {corpus.label} contain nonnumeric {col} values. Examples:\n"
                + sample.to_string(index=False)
            )
        data[col] = converted

    dupes = data.duplicated(subset=["text_id", "variant_id"], keep=False)
    if dupes.any():
        sample = data.loc[dupes, ["text_id", "title", "metric_label", "resource_label", "profile_label"]].head(10)
        raise RuntimeError(
            f"Duplicate exact metric rows were found in {corpus.label}; refusing to guess which row to use.\n"
            + sample.to_string(index=False)
        )

    data["corpus"] = corpus.label
    data["source_file"] = corpus.path.name
    return data


# ---------------------------------------------------------------------------
# Calculations
# ---------------------------------------------------------------------------


def quantile_or_nan(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return math.nan
    return float(np.quantile(values, q))


def descriptive_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "mean": math.nan,
            "median": math.nan,
            "minimum": math.nan,
            "maximum": math.nan,
            "within_corpus_range": math.nan,
            "standard_deviation": math.nan,
            "q1": math.nan,
            "q3": math.nan,
            "iqr": math.nan,
        }
    minimum = float(values.min())
    maximum = float(values.max())
    q1 = quantile_or_nan(values, 0.25)
    q3 = quantile_or_nan(values, 0.75)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "minimum": minimum,
        "maximum": maximum,
        "within_corpus_range": maximum - minimum,
        "standard_deviation": float(values.std(ddof=0)),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
    }


def coverage_status(retention: float) -> str:
    if not math.isfinite(retention):
        return "unavailable"
    if retention >= 0.90:
        return "high retention"
    if retention >= 0.75:
        return "moderate retention"
    return "low retention"


def build_analysis_tables(
    corpora: Sequence[CorpusInput],
    variants: Sequence[ComparisonVariant],
    loaded: dict[str, pd.DataFrame],
    *,
    coverage_threshold: Optional[float],
    bootstrap_resamples: int,
    bootstrap_confidence: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    variant_lookup = {v.variant_id: v for v in variants}
    corpus_result_rows: list[dict[str, object]] = []
    work_rows: list[pd.DataFrame] = []
    values_lookup: dict[tuple[str, str], np.ndarray] = {}

    for corpus in corpora:
        data = loaded[corpus.label].copy()
        for variant in variants:
            rows = data.loc[data["variant_id"] == variant.variant_id].copy()
            rows["qualifies"] = rows["value"].notna()
            if coverage_threshold is not None:
                rows["qualifies"] &= rows["coverage"].notna() & (rows["coverage"] >= coverage_threshold)
            qualifying = rows.loc[rows["qualifies"]].copy()
            vals = qualifying["value"].to_numpy(dtype=float)
            values_lookup[(corpus.label, variant.variant_id)] = vals
            stats = descriptive_stats(vals)
            n_total = int(corpus.report.work_count)
            n_eligible = int(vals.size)
            retention = (n_eligible / n_total) if n_total else math.nan
            corpus_result_rows.append({
                "corpus": corpus.label,
                "source_file": corpus.path.name,
                "metric": variant.metric_label,
                "resource": variant.resource_label,
                "analysis_view": variant.analysis_view,
                "weighting": variant.weighting,
                "profile": variant.profile_label,
                "scale": variant.scale,
                "eligible_works": n_eligible,
                "total_works": n_total,
                "excluded_works": n_total - n_eligible,
                "retention_fraction": retention,
                "retention_percent": retention * 100.0 if math.isfinite(retention) else math.nan,
                "coverage_status": coverage_status(retention),
                **stats,
                "variant_id": variant.variant_id,
            })
            if not rows.empty:
                keep = [
                    "corpus", "source_file", "text_id", "title", "author", "collection",
                    "metric_label", "resource_label", "analysis_view", "weighting", "profile_label",
                    "scale", "value", "coverage", "observations", "matched_tokens", "lexical_tokens",
                    "variant_id", "qualifies",
                ]
                work_rows.append(rows[keep])

    corpus_results = pd.DataFrame(corpus_result_rows)
    work_values = pd.concat(work_rows, ignore_index=True) if work_rows else pd.DataFrame()

    # Across-corpus summary for each exact metric/resource/profile.
    across_rows: list[dict[str, object]] = []
    for variant in variants:
        group = corpus_results.loc[corpus_results["variant_id"] == variant.variant_id].copy()
        finite = group.loc[group["mean"].notna()].copy()
        if finite.empty:
            continue
        low_row = finite.loc[finite["mean"].idxmin()]
        high_row = finite.loc[finite["mean"].idxmax()]
        across_rows.append({
            "metric": variant.metric_label,
            "resource": variant.resource_label,
            "profile": variant.profile_label,
            "scale": variant.scale,
            "lowest_corpus_mean": float(low_row["mean"]),
            "lowest_mean_corpus": str(low_row["corpus"]),
            "highest_corpus_mean": float(high_row["mean"]),
            "highest_mean_corpus": str(high_row["corpus"]),
            "across_corpus_mean_range": float(high_row["mean"] - low_row["mean"]),
            "corpora_compared": int(len(finite)),
            "variant_id": variant.variant_id,
        })
    across_summary = pd.DataFrame(across_rows)

    # Pairwise comparisons. Each pair uses its own independently qualifying work sets.
    pair_rows: list[dict[str, object]] = []
    seed_sequence = np.random.SeedSequence(random_seed)
    pair_jobs = len(list(combinations(corpora, 2))) * len(variants)
    child_seeds = iter(seed_sequence.spawn(max(pair_jobs, 1)))
    for corpus_a, corpus_b in combinations(corpora, 2):
        pair_label = f"{corpus_a.label} - {corpus_b.label}"
        for variant in variants:
            a = values_lookup[(corpus_a.label, variant.variant_id)]
            b = values_lookup[(corpus_b.label, variant.variant_id)]
            stats_a = descriptive_stats(a)
            stats_b = descriptive_stats(b)
            mean_diff = (
                float(stats_a["mean"] - stats_b["mean"])
                if math.isfinite(stats_a["mean"]) and math.isfinite(stats_b["mean"])
                else math.nan
            )
            median_diff = (
                float(stats_a["median"] - stats_b["median"])
                if math.isfinite(stats_a["median"]) and math.isfinite(stats_b["median"])
                else math.nan
            )
            delta = cliff_delta(a, b)
            rng = np.random.default_rng(next(child_seeds))
            ci_low, ci_high = bootstrap_mean_difference_ci(
                a,
                b,
                n_resamples=bootstrap_resamples,
                confidence=bootstrap_confidence,
                rng=rng,
            )
            if math.isfinite(mean_diff):
                if mean_diff > 0:
                    mean_direction = f"{corpus_a.label} corpus higher"
                elif mean_diff < 0:
                    mean_direction = f"{corpus_b.label} corpus higher"
                else:
                    mean_direction = "equal means"
            else:
                mean_direction = "unavailable"
            if math.isfinite(delta):
                if delta > 0:
                    dominance = f"{corpus_a.label} corpus tends higher"
                elif delta < 0:
                    dominance = f"{corpus_b.label} corpus tends higher"
                else:
                    dominance = "no directional dominance"
            else:
                dominance = "unavailable"

            n_a = int(a.size)
            n_b = int(b.size)
            total_a = int(corpus_a.report.work_count)
            total_b = int(corpus_b.report.work_count)
            pair_rows.append({
                "corpus_a": corpus_a.label,
                "corpus_b": corpus_b.label,
                "pair": pair_label,
                "metric": variant.metric_label,
                "resource": variant.resource_label,
                "profile": variant.profile_label,
                "analysis_view": variant.analysis_view,
                "weighting": variant.weighting,
                "scale": variant.scale,
                "n_a": n_a,
                "n_b": n_b,
                "total_works_a": total_a,
                "total_works_b": total_b,
                "retention_percent_a": (100.0 * n_a / total_a) if total_a else math.nan,
                "retention_percent_b": (100.0 * n_b / total_b) if total_b else math.nan,
                "mean_a": stats_a["mean"],
                "mean_b": stats_b["mean"],
                "mean_difference_a_minus_b": mean_diff,
                "mean_difference_display": signed_display(mean_diff),
                "absolute_mean_difference": abs(mean_diff) if math.isfinite(mean_diff) else math.nan,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "bootstrap_confidence": bootstrap_confidence,
                "mean_direction": mean_direction,
                "median_a": stats_a["median"],
                "median_b": stats_b["median"],
                "median_difference_a_minus_b": median_diff,
                "median_difference_display": signed_display(median_diff),
                "cliffs_delta": delta,
                "absolute_cliffs_delta": abs(delta) if math.isfinite(delta) else math.nan,
                "cliffs_delta_magnitude": cliff_magnitude(delta),
                "cliffs_delta_direction": dominance,
                "variant_id": variant.variant_id,
            })

    pairwise = pd.DataFrame(pair_rows)
    if not pairwise.empty:
        pairwise["rank_within_pair_by_effect"] = (
            pairwise.groupby("pair")["absolute_cliffs_delta"]
            .rank(method="min", ascending=False, na_option="bottom")
            .astype("Int64")
        )
        pairwise["rank_overall_by_effect"] = (
            pairwise["absolute_cliffs_delta"]
            .rank(method="min", ascending=False, na_option="bottom")
            .astype("Int64")
        )
        largest = pairwise.sort_values(
            ["absolute_cliffs_delta", "pair", "metric"],
            ascending=[False, True, True],
            kind="stable",
            na_position="last",
        ).reset_index(drop=True)
    else:
        largest = pairwise.copy()

    # Metric-level distinguishing summary. This prevents a metric selected under several
    # profiles/resources from occupying several adjacent rows in the cross-metric ranking.
    metric_summary_rows: list[dict[str, object]] = []
    if not pairwise.empty:
        for (pair_name, metric_name), group in pairwise.groupby(["pair", "metric"], sort=False):
            usable = group.loc[group["absolute_cliffs_delta"].notna()].copy()
            if usable.empty:
                continue
            strongest = usable.loc[usable["absolute_cliffs_delta"].idxmax()]
            weakest = usable.loc[usable["absolute_cliffs_delta"].idxmin()]
            deltas = usable["cliffs_delta"].astype(float)
            nonzero_signs = set(np.sign(deltas.loc[deltas != 0]).astype(int).tolist())
            if len(nonzero_signs) <= 1:
                direction_consistency = "consistent"
            else:
                direction_consistency = "direction changes across selected variants"
            metric_summary_rows.append({
                "pair": pair_name,
                "metric": metric_name,
                "variants_compared": int(len(usable)),
                "strongest_absolute_cliffs_delta": float(strongest["absolute_cliffs_delta"]),
                "strongest_cliffs_delta": float(strongest["cliffs_delta"]),
                "strongest_magnitude": str(strongest["cliffs_delta_magnitude"]),
                "strongest_direction": str(strongest["cliffs_delta_direction"]),
                "strongest_resource": str(strongest["resource"]),
                "strongest_profile": str(strongest["profile"]),
                "weakest_absolute_cliffs_delta": float(weakest["absolute_cliffs_delta"]),
                "weakest_resource": str(weakest["resource"]),
                "weakest_profile": str(weakest["profile"]),
                "mean_absolute_cliffs_delta_across_variants": float(usable["absolute_cliffs_delta"].mean()),
                "effect_size_span_across_variants": float(usable["absolute_cliffs_delta"].max() - usable["absolute_cliffs_delta"].min()),
                "direction_consistency": direction_consistency,
            })
    metric_summary = pd.DataFrame(metric_summary_rows)
    if not metric_summary.empty:
        metric_summary["rank_within_pair"] = (
            metric_summary.groupby("pair")["strongest_absolute_cliffs_delta"]
            .rank(method="min", ascending=False)
            .astype("Int64")
        )
        metric_summary = metric_summary.sort_values(
            ["pair", "rank_within_pair", "metric"], kind="stable"
        ).reset_index(drop=True)

    return corpus_results, across_summary, pairwise, largest, metric_summary, work_values


# ---------------------------------------------------------------------------
# Minimal XLSX writer (standard library only)
# ---------------------------------------------------------------------------


def _excel_col(index: int) -> str:
    letters = ""
    n = index
    while n:
        n, rem = divmod(n - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def _clean_xml_text(value: object) -> str:
    text = str(value if value is not None else "")
    text = "".join(ch for ch in text if ch in "\t\n\r" or ord(ch) >= 32)
    return xml_escape(text, {'"': '&quot;'})


def _cell_xml(row: int, col: int, value: object, style_id: int = 0) -> str:
    ref = f"{_excel_col(col)}{row}"
    if isinstance(value, StyledCell):
        style_id = value.style_id
        value = value.value
    style = f' s="{style_id}"' if style_id else ""
    if value is None:
        return f'<c r="{ref}"{style}/>'
    if isinstance(value, (bool, np.bool_)):
        return f'<c r="{ref}" t="b"{style}><v>{1 if bool(value) else 0}</v></c>'
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            rendered = str(int(number)) if number.is_integer() else repr(number)
            return f'<c r="{ref}"{style}><v>{rendered}</v></c>'
        return f'<c r="{ref}"{style}/>'
    text = _clean_xml_text(value)
    return f'<c r="{ref}" t="inlineStr"{style}><is><t xml:space="preserve">{text}</t></is></c>'


def _worksheet_xml(rows: Sequence[Sequence[object]], *, freeze_header: bool = True) -> str:
    max_cols = max((len(r) for r in rows), default=1)
    row_xml: list[str] = []
    for r_idx, values in enumerate(rows, start=1):
        cells = "".join(_cell_xml(r_idx, c_idx, value) for c_idx, value in enumerate(values, start=1))
        row_xml.append(f'<row r="{r_idx}">{cells}</row>')
    pane = (
        '<sheetViews><sheetView workbookViewId="0">'
        '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
        '</sheetView></sheetViews>'
        if freeze_header and rows
        else '<sheetViews><sheetView workbookViewId="0"/></sheetViews>'
    )
    dimension = f"A1:{_excel_col(max_cols)}{max(len(rows), 1)}"
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="{dimension}"/>{pane}'
        '<sheetFormatPr defaultRowHeight="15"/>'
        f'<sheetData>{"".join(row_xml)}</sheetData>'
        '<autoFilter ref="A1:' + _excel_col(max_cols) + '1"/>'
        '</worksheet>'
    )


def _styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="4">'
        '<font><sz val="11"/><name val="Calibri"/></font>'
        '<font><b/><sz val="11"/><name val="Calibri"/></font>'
        '<font><b/><sz val="14"/><name val="Calibri"/></font>'
        '<font><b/><sz val="11"/><name val="Calibri"/></font>'
        '</fonts>'
        '<fills count="4">'
        '<fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FFEAF2F8"/><bgColor indexed="64"/></patternFill></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FFF3F6F8"/><bgColor indexed="64"/></patternFill></fill>'
        '</fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="4">'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1" applyAlignment="1"><alignment wrapText="1"/></xf>'
        '<xf numFmtId="0" fontId="2" fillId="0" borderId="0" xfId="0" applyFont="1"/>'
        '<xf numFmtId="0" fontId="3" fillId="3" borderId="0" xfId="0" applyFont="1" applyFill="1"/>'
        '</cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        '</styleSheet>'
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


def write_simple_xlsx(path: Path, sheets: Sequence[tuple[str, list[list[object]], bool]]) -> None:
    if not sheets:
        raise ValueError("At least one worksheet is required.")
    used: set[str] = set()
    normalized = [(_sanitize_sheet_name(name, used), rows, freeze) for name, rows, freeze in sheets]
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
        for idx, (name, _, _) in enumerate(normalized, start=1)
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{sheets_xml}</sheets></workbook>'
    )
    rels = "".join(
        f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>'
        for idx in range(1, len(normalized) + 1)
    )
    rels += (
        f'<Relationship Id="rId{len(normalized)+1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    )
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
        for idx, (_, rows, freeze) in enumerate(normalized, start=1):
            archive.writestr(f"xl/worksheets/sheet{idx}.xml", _worksheet_xml(rows, freeze_header=freeze))


def frame_rows(frame: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> list[list[object]]:
    if frame is None or frame.empty:
        return [["No rows available for this selection."]]
    view = frame.loc[:, list(columns)] if columns is not None else frame
    rows: list[list[object]] = [[StyledCell(str(col), 1) for col in view.columns]]
    for record in view.itertuples(index=False, name=None):
        rows.append([json_ready(v) for v in record])
    return rows


def metadata_rows(metadata: dict[str, object]) -> list[list[object]]:
    rows: list[list[object]] = [[StyledCell("field", 1), StyledCell("value", 1)]]
    for key, value in metadata.items():
        rendered = (
            json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True)
            if isinstance(value, (dict, list, tuple))
            else json_ready(value)
        )
        rows.append([key, rendered])
    return rows


def corpus_readme_rows(metadata: dict[str, object]) -> list[list[object]]:
    return [
        [StyledCell("VerseVAD Corpus Comparison: Descriptive Results", 2), ""],
        ["Question", "What does each corpus look like?"],
        ["Unit of analysis", "Each qualifying work contributes one VerseVAD work-level value; corpus means are equal-work means."],
        ["Comparison discipline", "Only identical metric + resource + scope + weighting combinations are compared across corpora."],
        ["Within-corpus range", "Maximum work value minus minimum work value inside that corpus."],
        ["Across-corpus mean range", "Highest selected corpus mean minus lowest selected corpus mean for the same exact measurement."],
        ["Standard deviation", "Population SD of observed work-level values in the selected corpus."],
        ["IQR", "75th percentile minus 25th percentile of work-level values."],
        ["Coverage threshold", metadata.get("coverage_threshold_display", "None")],
        ["Created", metadata.get("created_at_local", "")],
    ]


def pairwise_readme_rows(metadata: dict[str, object]) -> list[list[object]]:
    confidence_pct = 100.0 * float(metadata.get("bootstrap_confidence", DEFAULT_BOOTSTRAP_CONFIDENCE))
    return [
        [StyledCell("VerseVAD Corpus Comparison: Pairwise Differences", 2), ""],
        ["Questions", "How far apart are these corpora? Which measured qualities distinguish them most strongly?"],
        ["Signed difference", "Always Corpus A minus Corpus B. Positive means Corpus A has the higher mean; negative means Corpus B has the higher mean."],
        ["Raw difference", "Meaningful within the same metric because both corpora use the same exported ruler."],
        ["Cliff's delta", "Scale-independent effect size. Positive means Corpus A work values tend to exceed Corpus B; negative means the reverse; zero indicates substantial balance/overlap."],
        ["Effect-size ranking", "Largest Differences is ranked by absolute Cliff's delta so metrics on different scales can be compared without treating their raw units as interchangeable."],
        ["Magnitude labels", "Descriptive guide only: |delta| < .147 negligible; < .330 small; < .474 medium; otherwise large."],
        ["Bootstrap CI", f"{confidence_pct:.1f}% percentile bootstrap confidence interval for the signed mean difference, based on independent work-level resampling within each corpus."],
        ["Formal hypothesis testing", "Not performed. This script intentionally reports estimation and effect size without p-values."],
        ["Percentage differences", "Not calculated. VerseVAD scales generally do not support meaningful statements such as '17% more' for these constructs."],
        ["Coverage threshold", metadata.get("coverage_threshold_display", "None")],
        ["Created", metadata.get("created_at_local", "")],
    ]


def matrix_rows(
    pairwise: pd.DataFrame,
    corpora: Sequence[CorpusInput],
    variants: Sequence[ComparisonVariant],
    *,
    value_column: str,
    title: str,
) -> list[list[object]]:
    labels = [c.label for c in corpora]
    rows: list[list[object]] = [[StyledCell(title, 2)]]
    for variant in variants:
        rows.extend([
            [],
            [StyledCell(variant.metric_label, 3)],
            ["Resource", variant.resource_label],
            ["Profile", variant.profile_label],
            ["Scale", variant.scale],
        ])
        rows.append([StyledCell("row minus column", 1)] + [StyledCell(label, 1) for label in labels])
        subset = pairwise.loc[pairwise["variant_id"] == variant.variant_id]
        lookup: dict[tuple[str, str], float] = {}
        for record in subset.itertuples(index=False):
            val = getattr(record, value_column)
            lookup[(str(record.corpus_a), str(record.corpus_b))] = float(val) if pd.notna(val) else math.nan
            lookup[(str(record.corpus_b), str(record.corpus_a))] = -float(val) if pd.notna(val) else math.nan
        for row_label in labels:
            row: list[object] = [row_label]
            for col_label in labels:
                if row_label == col_label:
                    row.append(0.0)
                else:
                    row.append(lookup.get((row_label, col_label), math.nan))
            rows.append(row)
    return rows


def write_outputs(
    run_folder: Path,
    corpora: Sequence[CorpusInput],
    variants: Sequence[ComparisonVariant],
    corpus_results: pd.DataFrame,
    across_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    largest: pd.DataFrame,
    metric_summary: pd.DataFrame,
    work_values: pd.DataFrame,
    metadata: dict[str, object],
    spec: dict[str, object],
) -> None:
    run_folder.mkdir(parents=True, exist_ok=True)

    coverage_cols = [
        "corpus", "metric", "resource", "profile", "eligible_works", "total_works",
        "excluded_works", "retention_percent", "coverage_status",
    ]
    coverage = corpus_results[coverage_cols].copy() if not corpus_results.empty else pd.DataFrame(columns=coverage_cols)

    corpus_results.to_csv(run_folder / "corpus_metric_results.csv", index=False)
    across_summary.to_csv(run_folder / "across_corpus_summary.csv", index=False)
    coverage.to_csv(run_folder / "coverage_summary.csv", index=False)
    pairwise.to_csv(run_folder / "pairwise_differences.csv", index=False)
    largest.to_csv(run_folder / "largest_differences.csv", index=False)
    metric_summary.to_csv(run_folder / "metric_distinguishing_summary.csv", index=False)
    if not pairwise.empty:
        effect_cols = [
            "corpus_a", "corpus_b", "pair", "metric", "resource", "profile",
            "cliffs_delta", "absolute_cliffs_delta", "cliffs_delta_magnitude",
            "cliffs_delta_direction", "rank_within_pair_by_effect", "rank_overall_by_effect",
        ]
        pairwise[effect_cols].to_csv(run_folder / "pairwise_effect_sizes.csv", index=False)
    else:
        pd.DataFrame().to_csv(run_folder / "pairwise_effect_sizes.csv", index=False)
    work_values.to_csv(run_folder / "work_values_long.csv", index=False)

    with (run_folder / "analysis_spec.json").open("w", encoding="utf-8") as handle:
        json.dump(json_ready(spec), handle, ensure_ascii=False, indent=2, sort_keys=True)
    with (run_folder / "analysis_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(json_ready(metadata), handle, ensure_ascii=False, indent=2, sort_keys=True)

    corpus_columns = [
        "corpus", "metric", "resource", "profile", "scale", "eligible_works", "total_works",
        "retention_percent", "mean", "median", "minimum", "maximum", "within_corpus_range",
        "standard_deviation", "q1", "q3", "iqr",
    ]
    across_columns = [
        "metric", "resource", "profile", "scale", "lowest_mean_corpus", "lowest_corpus_mean",
        "highest_mean_corpus", "highest_corpus_mean", "across_corpus_mean_range", "corpora_compared",
    ]
    corpus_sheets: list[tuple[str, list[list[object]], bool]] = [
        ("00 Read Me", corpus_readme_rows(metadata), False),
        ("01 Corpus Overview", frame_rows(corpus_results, corpus_columns), True),
        ("02 Across Corpus Range", frame_rows(across_summary, across_columns), True),
        ("03 Coverage", frame_rows(coverage), True),
    ]
    metric_order = list(dict.fromkeys(v.metric_label for v in variants))
    for idx, metric_name in enumerate(metric_order, start=1):
        detail = corpus_results.loc[corpus_results["metric"] == metric_name, corpus_columns]
        corpus_sheets.append((f"M{idx:02d} {metric_name}", frame_rows(detail), True))
    corpus_sheets.append(("Run Metadata", metadata_rows(metadata), True))
    write_simple_xlsx(run_folder / "corpus_results.xlsx", corpus_sheets)

    pair_columns = [
        "corpus_a", "corpus_b", "metric", "resource", "profile", "scale", "n_a", "n_b",
        "mean_a", "mean_b", "mean_difference_a_minus_b", "mean_difference_display",
        "bootstrap_ci_low", "bootstrap_ci_high", "median_a", "median_b",
        "median_difference_a_minus_b", "cliffs_delta", "absolute_cliffs_delta",
        "cliffs_delta_magnitude", "cliffs_delta_direction", "rank_within_pair_by_effect",
    ]
    largest_columns = [
        "rank_overall_by_effect", "pair", "metric", "resource", "profile", "mean_a", "mean_b",
        "mean_difference_a_minus_b", "mean_difference_display", "cliffs_delta",
        "absolute_cliffs_delta", "cliffs_delta_magnitude", "cliffs_delta_direction",
        "bootstrap_ci_low", "bootstrap_ci_high", "n_a", "n_b",
    ]
    metric_summary_columns = [
        "pair", "rank_within_pair", "metric", "variants_compared",
        "strongest_absolute_cliffs_delta", "strongest_cliffs_delta", "strongest_magnitude",
        "strongest_direction", "strongest_resource", "strongest_profile",
        "weakest_absolute_cliffs_delta", "mean_absolute_cliffs_delta_across_variants",
        "effect_size_span_across_variants", "direction_consistency",
    ]
    pair_sheets: list[tuple[str, list[list[object]], bool]] = [
        ("00 Read Me", pairwise_readme_rows(metadata), False),
        ("01 Pairwise Summary", frame_rows(pairwise, pair_columns), True),
        ("02 Largest Differences", frame_rows(largest, largest_columns), True),
        ("03 Metric Rankings", frame_rows(metric_summary, metric_summary_columns), True),
    ]
    for idx, (a, b) in enumerate(combinations([c.label for c in corpora], 2), start=1):
        detail = pairwise.loc[(pairwise["corpus_a"] == a) & (pairwise["corpus_b"] == b), pair_columns]
        pair_sheets.append((f"P{idx:02d} {a} - {b}", frame_rows(detail), True))
    pair_sheets.append((
        "Mean Difference Matrices",
        matrix_rows(pairwise, corpora, variants, value_column="mean_difference_a_minus_b", title="Signed mean differences"),
        False,
    ))
    pair_sheets.append((
        "Effect Size Matrices",
        matrix_rows(pairwise, corpora, variants, value_column="cliffs_delta", title="Cliff's delta"),
        False,
    ))
    pair_sheets.append(("Run Metadata", metadata_rows(metadata), True))
    write_simple_xlsx(run_folder / "pairwise_differences.xlsx", pair_sheets)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare 2-5 VerseVAD corpora using exact like-for-like metric variants.")
    parser.add_argument("--version", action="version", version=f"compare.py {__version__}")
    parser.add_argument("--source-dir", type=Path, default=None, help="Folder containing VerseVAD corpus exports.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Root output folder. Default: <project>/exports/compare_corpus")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_console_encoding()
    args = _build_parser().parse_args(argv)
    root = project_root()
    source_dir = (args.source_dir or (root / "source")).expanduser().resolve()
    output_root = (args.output_dir or (root / "exports" / "compare_corpus")).expanduser().resolve()

    print("VerseVAD Corpus Comparison")
    print("==========================")
    print("Questions:")
    print("  1. What does each corpus look like?")
    print("  2. How far apart are these corpora?")
    print("  3. Which measured qualities distinguish them most strongly?")
    print("\nLike-for-like rule: metric, resource, scope, weighting, and scale must match across corpora.")

    sources = discover_sources(source_dir)
    selected_paths = choose_sources(sources)
    labels = maybe_customize_labels(selected_paths)
    corpora = validate_inputs(selected_paths, labels)

    concepts = build_common_metric_concepts(corpora)
    print(f"\nCompatible metric concepts across all selected corpora: {len(concepts)}")
    selected_metrics = choose_metrics(concepts)
    resource_choices = choose_resources(corpora, selected_metrics)
    profiles = common_profiles(corpora, selected_metrics, resource_choices)
    selected_profiles = choose_profiles(profiles)
    variants = build_comparison_variants(corpora, selected_metrics, resource_choices, selected_profiles)

    print("\n# Coverage")
    print("Apply a minimum work-level VerseVAD coverage threshold uniformly to every corpus.")
    print("Examples: 80 or 0.80. Press Enter for no additional threshold.")
    while True:
        raw = input("Minimum coverage: ").strip()
        try:
            coverage_threshold = parse_coverage_threshold(raw)
            break
        except ValueError as exc:
            print(str(exc))

    print("\n# Bootstrap confidence intervals")
    print("Pairwise mean differences use independent work-level bootstrap resampling within each corpus.")
    raw = input(f"Bootstrap resamples [{DEFAULT_BOOTSTRAP_RESAMPLES}]: ").strip()
    if raw:
        try:
            bootstrap_resamples = int(raw)
            if bootstrap_resamples < 100:
                raise ValueError
        except ValueError:
            raise SystemExit("Bootstrap resamples must be an integer of at least 100.")
    else:
        bootstrap_resamples = DEFAULT_BOOTSTRAP_RESAMPLES

    print("\n# Analysis receipt")
    print("Corpora:")
    for corpus in corpora:
        print(f"  - {corpus.label}: {corpus.report.work_count:,} works")
    print("Metrics:")
    for concept in selected_metrics:
        chosen = resource_choices[concept.key]
        labels_for_resources = []
        for rid in chosen:
            label = next((v.resource_label for v in variants if v.metric_key == concept.key and v.lexicon_id == rid), rid)
            labels_for_resources.append(label)
        print(f"  - {concept.label}: {', '.join(labels_for_resources)}")
    print("Profiles:")
    for p in selected_profiles:
        print(f"  - {profile_label(*p)}")
    print(f"Exact comparison variants: {len(variants)}")
    print(
        "Coverage threshold: "
        + (f"{coverage_threshold*100:.1f}%" if coverage_threshold is not None else "None")
    )
    print(f"Bootstrap: {bootstrap_resamples:,} resamples · {DEFAULT_BOOTSTRAP_CONFIDENCE*100:.0f}% CI · seed {DEFAULT_RANDOM_SEED}")
    print("Formal hypothesis testing: OFF")
    raw = input("\nRun comparison? [Y/n]: ").strip().casefold()
    if raw in {"n", "no"}:
        print("Cancelled.")
        return 0

    print("\nLoading selected VerseVAD measurements...")
    loaded: dict[str, pd.DataFrame] = {}
    for corpus in corpora:
        print(f"  {corpus.label}...")
        loaded[corpus.label] = load_selected_variants(corpus, variants)

    print("Calculating corpus descriptions and pairwise differences...")
    corpus_results, across_summary, pairwise, largest, metric_summary, work_values = build_analysis_tables(
        corpora,
        variants,
        loaded,
        coverage_threshold=coverage_threshold,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_confidence=DEFAULT_BOOTSTRAP_CONFIDENCE,
        random_seed=DEFAULT_RANDOM_SEED,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = "_vs_".join(slugify(c.label)[:24] for c in corpora)
    run_folder = output_root / f"{run_name}_{stamp}"
    counter = 2
    while run_folder.exists():
        run_folder = output_root / f"{run_name}_{stamp}_{counter}"
        counter += 1

    created_at = datetime.now().astimezone().isoformat()
    coverage_display = f"{coverage_threshold*100:.1f}%" if coverage_threshold is not None else "None"
    spec = {
        "program": "compare.py",
        "version": __version__,
        "corpora": [
            {"label": c.label, "source": str(c.path), "work_count": int(c.report.work_count)} for c in corpora
        ],
        "metrics": [asdict(c) for c in selected_metrics],
        "resource_choices": resource_choices,
        "profiles": [{"analysis_view": p[0], "weighting": p[1], "label": profile_label(*p)} for p in selected_profiles],
        "variants": [asdict(v) for v in variants],
        "coverage_threshold": coverage_threshold,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_confidence": DEFAULT_BOOTSTRAP_CONFIDENCE,
        "random_seed": DEFAULT_RANDOM_SEED,
        "hypothesis_testing": False,
        "difference_direction": "Corpus A minus Corpus B",
        "cross_metric_ranking": "absolute Cliff's delta",
    }
    metadata = {
        "program": "compare.py",
        "version": __version__,
        "created_at_local": created_at,
        "source_directory": str(source_dir),
        "corpus_count": len(corpora),
        "corpus_labels": [c.label for c in corpora],
        "source_files": [c.path.name for c in corpora],
        "source_sha256": {c.label: file_sha256(c.path) for c in corpora},
        "source_schema_fingerprints": {c.label: c.report.schema_fingerprint for c in corpora},
        "work_counts": {c.label: int(c.report.work_count) for c in corpora},
        "selected_metric_count": len(selected_metrics),
        "exact_variant_count": len(variants),
        "coverage_threshold": coverage_threshold,
        "coverage_threshold_display": coverage_display,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_confidence": DEFAULT_BOOTSTRAP_CONFIDENCE,
        "random_seed": DEFAULT_RANDOM_SEED,
        "hypothesis_testing": False,
        "corpus_aggregation": "equal-work mean of work-level VerseVAD values",
        "difference_definition": "Corpus A minus Corpus B",
        "effect_size": "Cliff's delta",
        "percentage_differences": False,
    }

    print("Writing Excel, CSV, and reproducibility outputs...")
    write_outputs(
        run_folder,
        corpora,
        variants,
        corpus_results,
        across_summary,
        pairwise,
        largest,
        metric_summary,
        work_values,
        metadata,
        spec,
    )

    print("\nComparison complete.")
    print(f"Output folder: {run_folder}")
    print("\nOpen first:")
    print("  corpus_results.xlsx       → What does each corpus look like?")
    print("  pairwise_differences.xlsx → How far apart are they, and what distinguishes them most?")
    print("\nNo hypothesis tests or percentage-difference claims were generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
