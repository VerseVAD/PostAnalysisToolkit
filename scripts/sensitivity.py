#!/usr/bin/env python3
"""sensitivity.py

Interactive methodological sensitivity analysis for VerseVAD corpus exports.

Designed to live beside ``versevad_reader.py`` inside::

    versevad_stats/
        scripts/
            versevad_reader.py
            correlation.py
            anomaly.py
            sensitivity.py
        source/
            <VerseVAD Complete Audit ZIPs or standalone corpus_vad_metrics.csv>
        exports/

Run from the ``versevad_stats`` project folder with::

    python scripts/sensitivity.py

Core questions
--------------
This program is deliberately organized around two research questions:

1. Corpus sensitivity
   How much do equal-work corpus means for selected metrics move when the same
   poems are measured under different reasonable VerseVAD scope/weighting
   profiles and, where the export contains multiple compatible normative
   resources, different resources?

2. Poem sensitivity
   For a specific poem, how much does the measurement and the poem's corpus-
   relative position move under those same methodological variants?

The common-set alignment mode is the default and recommended mode. In common
mode, every methodological variant for a selected metric is compared on the
same qualifying poems. Pairwise/available mode is retained as an explicit
alternative when preserving more observations is more important than holding
sample composition constant.

Resource behavior
-----------------
The program analyzes VerseVAD's exported poem-level measurements. It does not
load Brysbaert, Lancaster, NRC, Warriner, SUBTLEX, Kuperman, or other source
lexicon files directly.

For metrics represented by one compatible resource in the VerseVAD export,
that resource is selected automatically. If the same construct is represented
by multiple resources (notably Valence, Arousal, and Dominance across NRC VAD
v1, NRC VAD v2.1, and Warriner), the terminal prompts the user to choose one or
more resources. The selection syntax is shared with profile selection:

    A          all choices
    1          choice 1
    1,2        choices 1 and 2
    1,3,5      choices 1, 3, and 5
    5-6        choices 5 through 6
    1,3,5-6    mixed individual choices and ranges

VerseVAD's VAD means are already exported on a normalized 0-1 scale for the
supported VAD resources, so this script uses those exported normalized values
as authoritative and never normalizes them a second time. Single-resource
metrics retain their native exported scales.

Principal outputs
-----------------
Each run writes to ``exports/sensitivity/<run>/``. Research-facing outputs are:

* sensitivity_summary.xlsx
    Compact corpus-level summary, exact corpus means by variant, top sensitive
    poems, most stable poems, selected-poem summaries, coverage/sample tables,
    pairwise agreement, metric details, and selected-poem detail sheets.

* sensitivity_poem_profiles.xlsx (optional, default yes)
    A Poem Index plus one worksheet per poem in the analyzed corpus. Each poem
    sheet contains sensitivity summaries and exact variant-level measurements
    for all selected metrics.

Audit/reproducibility outputs include:

* sensitivity_summary.csv
* corpus_sensitivity.csv
* coverage_sample.csv
* top_sensitive_poems.csv
* most_stable_poems.csv
* selected_poem_sensitivity.csv
* selected_poem_variants.csv
* pairwise_agreement.csv
* poem_sensitivity.csv
* variant_values_long.csv
* analysis_spec.json
* analysis_metadata.json

Dependencies
------------
Python 3.10+, numpy, pandas, scipy, and ``versevad_reader.py``.
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
from pathlib import Path
from typing import Optional, Sequence
from xml.sax.saxutils import escape as xml_escape

try:
    import numpy as np
    import pandas as pd
    from scipy.stats import rankdata, spearmanr
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "sensitivity.py requires numpy, pandas, and scipy.\n"
        "Install them with:\n"
        "  python -m pip install numpy pandas scipy"
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

try:
    import versevad_reader
    from versevad_reader import VerseVADCorpusReader
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Could not import versevad_reader.py. Put sensitivity.py beside "
        "versevad_reader.py inside the scripts folder."
    ) from exc


__version__ = "0.2.0"

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


@dataclass(frozen=True)
class MetricConcept:
    label: str
    metric: str
    dimension: str
    category: str
    resource_ids: tuple[str, ...]
    resource_labels: tuple[str, ...]
    profiles: tuple[tuple[str, str], ...]
    scales: tuple[str, ...]

    @property
    def key(self) -> str:
        return "|".join((self.metric, self.dimension, self.category))

    @property
    def search_blob(self) -> str:
        parts = [
            self.label,
            self.metric,
            self.dimension,
            self.category,
            *self.resource_ids,
            *self.resource_labels,
            *self.scales,
        ]
        return " ".join(str(part) for part in parts if str(part).strip()).casefold()


@dataclass(frozen=True)
class Variant:
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
    def profile_label(self) -> str:
        return profile_label(self.analysis_view, self.weighting)

    @property
    def resource_label(self) -> str:
        return resource_label(self.lexicon_id, self.lexicon)

    @property
    def label(self) -> str:
        return f"{self.resource_label} · {self.profile_label}"

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


@dataclass(frozen=True)
class Formula:
    expression: str
    display: str = ""


@dataclass(frozen=True)
class StyledCell:
    value: object
    style_id: int


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def project_root() -> Path:
    return shared_project_root(__file__)


def source_directory(root: Path) -> Path:
    return root / "source"


def export_directory(root: Path) -> Path:
    return root / "exports" / "sensitivity"


def slugify(text: str) -> str:
    return shared_slugify(text, fallback="analysis", max_length=72)


def discover_sources(directory: Path) -> list[Path]:
    return discover_corpus_metric_sources(directory, versevad_reader.METRICS_FILENAME)


def select_source(paths: Sequence[Path]) -> Path:
    if not paths:
        raise SystemExit(
            "No VerseVAD corpus sources were found in the source folder.\n"
            "Add a Complete Audit ZIP or standalone corpus_vad_metrics.csv and try again."
        )
    if len(paths) == 1:
        print("\nFound one corpus source:")
        print(paths[0].name)
        return paths[0]

    print("\nAvailable corpus sources:")
    for idx, path in enumerate(paths, start=1):
        print(f"[{idx}] {path.name}")
    while True:
        raw = input("\nSelect corpus: ").strip()
        try:
            choice = int(raw)
            if 1 <= choice <= len(paths):
                return paths[choice - 1]
        except ValueError:
            pass
        print("Enter one of the listed numbers.")


def safe_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def fmt_num(value: object, digits: int = 3) -> str:
    x = safe_float(value)
    if math.isnan(x):
        return "N/A"
    return f"{x:.{digits}f}"


def mean_pairwise_absolute(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if len(vals) < 2:
        return math.nan
    diffs = [abs(vals[i] - vals[j]) for i in range(len(vals)) for j in range(i + 1, len(vals))]
    return float(np.mean(diffs)) if diffs else math.nan


def largest_pairwise_difference(
    values: pd.Series,
    lookup: dict[str, Variant],
) -> tuple[float, str, str, str]:
    available = values.dropna()
    if len(available) < 2:
        return math.nan, "", "", ""
    best = (-1.0, "", "", "")
    ids = [str(v) for v in available.index]
    for i, left_id in enumerate(ids):
        for right_id in ids[i + 1 :]:
            diff = abs(float(available[left_id]) - float(available[right_id]))
            if diff > best[0]:
                best = (
                    diff,
                    lookup[left_id].label,
                    lookup[right_id].label,
                    comparison_type(lookup[left_id], lookup[right_id]),
                )
    return best


def comparison_type(left: Variant, right: Variant) -> str:
    changes: list[str] = []
    if left.lexicon_id != right.lexicon_id:
        changes.append("resource")
    if left.analysis_view != right.analysis_view:
        changes.append("scope")
    if left.weighting != right.weighting:
        changes.append("weighting")
    if not changes:
        return "identical method"
    return " + ".join(changes) + " change"


# ---------------------------------------------------------------------------
# Catalog construction and interactive selection
# ---------------------------------------------------------------------------


def build_metric_concepts(catalog: pd.DataFrame) -> list[MetricConcept]:
    work = catalog.copy().fillna("")
    concepts: list[MetricConcept] = []
    group_cols = ["metric", "dimension", "category"]
    for (metric, dimension, category), group in work.groupby(group_cols, sort=False, dropna=False):
        representative = group.iloc[0]
        resources = (
            group[["lexicon_id", "lexicon"]]
            .drop_duplicates()
            .sort_values(["lexicon", "lexicon_id"], kind="stable")
        )
        profiles = sorted(
            {(str(row.analysis_view), str(row.weighting)) for row in group.itertuples()},
            key=lambda p: (
                list(VIEW_LABELS).index(p[0]) if p[0] in VIEW_LABELS else 99,
                list(WEIGHTING_LABELS).index(p[1]) if p[1] in WEIGHTING_LABELS else 99,
            ),
        )
        resource_ids = tuple(resources["lexicon_id"].astype(str).tolist())
        resource_labels = tuple(
            resource_label(str(row.lexicon_id), str(row.lexicon)) for row in resources.itertuples()
        )
        scales = tuple(sorted({str(v) for v in group["scale"].tolist() if str(v).strip()}))
        concepts.append(
            MetricConcept(
                label=friendly_metric_label(representative),
                metric=str(metric),
                dimension=str(dimension),
                category=str(category),
                resource_ids=resource_ids,
                resource_labels=resource_labels,
                profiles=tuple(profiles),
                scales=scales,
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
    selected: list[MetricConcept] = []
    selected_keys: set[str] = set()
    print("\n# Select sensitivity metrics")
    print(
        "Search by ordinary terms such as: concreteness, frequency, aoa, interoceptive, "
        "arousal, valence, emotion intensity, or SD."
    )
    print("You may add as many metrics as you want. Type Q when finished.\n")

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
            print("No matching metrics. Try broader search terms.\n")
            continue
        shown = matches[:25]
        print(f"\nMatches ({len(matches)} total):")
        for idx, concept in enumerate(shown, start=1):
            resources = ", ".join(concept.resource_labels)
            print(f"[{idx}] {concept.label}")
            print(
                f"    metric={concept.metric}; dimension={concept.dimension or '—'}; "
                f"category={concept.category or '—'}; resources={resources}"
            )
        if len(matches) > len(shown):
            print(f"    ... {len(matches) - len(shown)} more match(es); refine the search to narrow them.")
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
        chosen = shown[idx - 1]
        if chosen.key in selected_keys:
            print(f"Already selected: {chosen.label}\n")
            continue
        selected.append(chosen)
        selected_keys.add(chosen.key)
        print(f"Added: {chosen.label}\n")


def resources_for_concept(catalog: pd.DataFrame, concept: MetricConcept) -> list[tuple[str, str]]:
    work = catalog.fillna("")
    rows = work.loc[
        (work["metric"].astype(str) == concept.metric)
        & (work["dimension"].astype(str) == concept.dimension)
        & (work["category"].astype(str) == concept.category),
        ["lexicon_id", "lexicon"],
    ].drop_duplicates()
    out = [(str(row.lexicon_id), str(row.lexicon)) for row in rows.itertuples()]
    return sorted(out, key=lambda item: resource_label(*item).casefold())


def choose_resources_per_metric(
    catalog: pd.DataFrame,
    concepts: Sequence[MetricConcept],
) -> dict[str, list[str]]:
    chosen: dict[str, list[str]] = {}
    print("\n# Compatible resources")
    print(
        "Resources are read from this VerseVAD export. Single-resource metrics are selected "
        "automatically; you are only asked when the same metric is represented by multiple resources."
    )
    for concept in concepts:
        resources = resources_for_concept(catalog, concept)
        if not resources:
            raise RuntimeError(f"No resource rows were found for {concept.label}.")
        if len(resources) == 1:
            rid, lexicon = resources[0]
            chosen[concept.key] = [rid]
            print(f"\n{concept.label}: {resource_label(rid, lexicon)} [automatic]")
            continue

        print(f"\n{concept.label} is available from multiple compatible resources:")
        for idx, (rid, lexicon) in enumerate(resources, start=1):
            print(f"[{idx}] {resource_label(rid, lexicon)}")
        print("Select one or more using A, 1,2, 1,3, 5-6, or combinations such as 1,3,5-6.")
        while True:
            raw = input("Selection: ").strip()
            try:
                indexes = parse_index_selection(raw, len(resources))
                chosen[concept.key] = [resources[idx][0] for idx in indexes]
                break
            except ValueError as exc:
                print(str(exc))
    return chosen


def relevant_profiles(
    catalog: pd.DataFrame,
    concepts: Sequence[MetricConcept],
    resource_choices: dict[str, list[str]],
) -> list[tuple[str, str]]:
    work = catalog.fillna("")
    mask = pd.Series(False, index=work.index)
    for concept in concepts:
        resource_set = set(resource_choices[concept.key])
        mask |= (
            (work["metric"].astype(str) == concept.metric)
            & (work["dimension"].astype(str) == concept.dimension)
            & (work["category"].astype(str) == concept.category)
            & (work["lexicon_id"].astype(str).isin(resource_set))
        )
    profiles = {(str(row.analysis_view), str(row.weighting)) for row in work.loc[mask].itertuples()}
    view_order = {view: idx for idx, view in enumerate(VIEW_LABELS)}
    weight_order = {w: idx for idx, w in enumerate(WEIGHTING_LABELS)}
    return sorted(profiles, key=lambda p: (view_order.get(p[0], 99), weight_order.get(p[1], 99)))


def choose_profiles(profiles: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    print("\n# Choose lexical scope / weighting profiles")
    for idx, item in enumerate(profiles, start=1):
        print(f"[{idx}] {profile_label(item[0], item[1])}")
    print("\nSelect using A, 1,2, 1,3, 5-6, or combinations such as 1,3,5-6.")
    print("Press Enter to include all available profiles.")
    while True:
        raw = input("Selection: ").strip()
        if not raw:
            return list(profiles)
        try:
            indexes = parse_index_selection(raw, len(profiles))
            return [profiles[idx] for idx in indexes]
        except ValueError as exc:
            print(str(exc))


def build_variants(
    catalog: pd.DataFrame,
    concepts: Sequence[MetricConcept],
    resource_choices: dict[str, list[str]],
    selected_profiles: Sequence[tuple[str, str]],
) -> list[Variant]:
    work = catalog.copy().fillna("")
    profile_set = set(selected_profiles)
    variants: list[Variant] = []
    seen: set[tuple[str, ...]] = set()

    for concept in concepts:
        resource_set = set(resource_choices[concept.key])
        rows = work.loc[
            (work["metric"].astype(str) == concept.metric)
            & (work["dimension"].astype(str) == concept.dimension)
            & (work["category"].astype(str) == concept.category)
        ].copy()
        for row in rows.itertuples():
            lexicon_id = str(row.lexicon_id)
            profile = (str(row.analysis_view), str(row.weighting))
            if lexicon_id not in resource_set or profile not in profile_set:
                continue
            exact_key = (
                lexicon_id,
                concept.metric,
                concept.dimension,
                concept.category,
                profile[0],
                profile[1],
            )
            if exact_key in seen:
                continue
            seen.add(exact_key)
            variants.append(
                Variant(
                    metric_key=concept.key,
                    metric_label=concept.label,
                    lexicon_id=lexicon_id,
                    lexicon=str(row.lexicon),
                    metric=concept.metric,
                    dimension=concept.dimension,
                    category=concept.category,
                    analysis_view=profile[0],
                    weighting=profile[1],
                    scale=str(row.scale),
                )
            )

    metric_order = {concept.key: idx for idx, concept in enumerate(concepts)}
    profile_order = {p: idx for idx, p in enumerate(selected_profiles)}
    variants.sort(
        key=lambda v: (
            metric_order.get(v.metric_key, 999),
            resource_label(v.lexicon_id, v.lexicon).casefold(),
            profile_order.get((v.analysis_view, v.weighting), 999),
        )
    )
    return variants


# ---------------------------------------------------------------------------
# Efficient batch loading through the VerseVAD reader
# ---------------------------------------------------------------------------


def _identity_string(frame: pd.DataFrame) -> pd.Series:
    cols = [frame[col].fillna("").astype(str) for col in IDENTITY_COLUMNS]
    out = cols[0]
    for col in cols[1:]:
        out = out + "\x1f" + col
    return out


def _variant_identity_string(variant: Variant) -> str:
    return "\x1f".join(variant.exact_key)


def load_selected_variants(reader: VerseVADCorpusReader, variants: Sequence[Variant]) -> pd.DataFrame:
    identity_to_variant = {_variant_identity_string(v): v for v in variants}
    wanted = set(identity_to_variant)
    pieces: list[pd.DataFrame] = []

    for chunk in reader._iter_chunks(usecols=LOAD_COLUMNS):  # intentional batch use of reader streaming layer
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
        raise RuntimeError("No rows matched the selected metric/resource/profile variants.")

    data = pd.concat(pieces, ignore_index=True)
    numeric_cols = ["value", "observations", "matched_tokens", "lexical_tokens", "coverage"]
    for col in numeric_cols:
        raw = data[col].astype(str).str.strip()
        converted = pd.to_numeric(raw.replace("", np.nan), errors="coerce")
        invalid = raw.ne("") & converted.isna()
        if invalid.any():
            sample = data.loc[
                invalid,
                ["title", "metric_label", "resource_label", "profile_label", col],
            ].head(5)
            raise RuntimeError(
                f"Selected data contain nonnumeric {col} values. Examples:\n{sample.to_string(index=False)}"
            )
        data[col] = converted

    dupes = data.duplicated(subset=["text_id", "variant_id"], keep=False)
    if dupes.any():
        sample = data.loc[
            dupes,
            ["text_id", "title", "metric_label", "resource_label", "profile_label"],
        ].head(10)
        raise RuntimeError(
            "Duplicate exact metric rows were found for the same poem/variant. "
            "Sensitivity analysis refuses to guess which row to use.\n"
            + sample.to_string(index=False)
        )

    eligible = data["lexical_tokens"]
    matched = data["matched_tokens"]
    recomputed = matched / eligible.replace(0, np.nan)
    comparable = data["coverage"].notna() & recomputed.notna()
    discrepancy = (data.loc[comparable, "coverage"] - recomputed.loc[comparable]).abs()
    if (discrepancy > 1e-10).any():
        raise RuntimeError(
            "Coverage metadata are internally inconsistent with matched_tokens / lexical_tokens."
        )

    data["matched_observations"] = data["matched_tokens"]
    data["eligible_observations"] = data["lexical_tokens"]
    data["observation_unit"] = (
        data["weighting"].map({"token": "token", "type": "type"}).fillna(data["weighting"])
    )
    return data


# ---------------------------------------------------------------------------
# Poem selection
# ---------------------------------------------------------------------------


def poem_catalog_from_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    cols = ["text_id", "title", "author", "collection", "date_label", "genre"]
    catalog = raw_data[cols].drop_duplicates(subset=["text_id"], keep="first").copy()
    catalog = catalog.sort_values(["title", "text_id"], kind="stable").reset_index(drop=True)
    return catalog


def search_poems(catalog: pd.DataFrame, query: str) -> pd.DataFrame:
    terms = [term.casefold() for term in re.findall(r"\S+", query.strip())]
    if not terms:
        return catalog.copy()
    blob = (
        catalog["title"].fillna("").astype(str)
        + " "
        + catalog["author"].fillna("").astype(str)
        + " "
        + catalog["text_id"].fillna("").astype(str)
    ).str.casefold()
    mask = pd.Series(True, index=catalog.index)
    for term in terms:
        mask &= blob.str.contains(re.escape(term), regex=True)
    return catalog.loc[mask].copy()


def choose_poems(catalog: pd.DataFrame) -> list[str]:
    selected_ids: list[str] = []
    selected_set: set[str] = set()
    print("\n# Select poems for detailed sensitivity output")
    print("Search by title words. Add as many poems as you want. Type Q when finished.\n")
    while True:
        if selected_ids:
            names = catalog.set_index("text_id").reindex(selected_ids)["title"].fillna("").tolist()
            print("Currently selected:")
            for idx, title in enumerate(names, start=1):
                print(f"  {idx}. {title}")
            print()
        query = input("Search poem (Q to finish): ").strip()
        if query.casefold() == "q":
            return selected_ids
        matches = search_poems(catalog, query)
        if matches.empty:
            print("No matching poems. Try broader title words.\n")
            continue
        shown = matches.head(25)
        print(f"\nMatches ({len(matches)} total):")
        for idx, row in enumerate(shown.itertuples(index=False), start=1):
            author = f" · {row.author}" if str(row.author).strip() else ""
            print(f"[{idx}] {row.title}{author}")
        if len(matches) > len(shown):
            print(f"    ... {len(matches) - len(shown)} more match(es); refine the search to narrow them.")
        raw = input("Select number, R to refine, or Q to finish: ").strip()
        if raw.casefold() == "r":
            print()
            continue
        if raw.casefold() == "q":
            return selected_ids
        try:
            idx = int(raw)
            if not 1 <= idx <= len(shown):
                raise ValueError
        except ValueError:
            print("Enter one of the displayed numbers, R, or Q.\n")
            continue
        text_id = str(shown.iloc[idx - 1]["text_id"])
        title = str(shown.iloc[idx - 1]["title"])
        if text_id in selected_set:
            print(f"Already selected: {title}\n")
            continue
        selected_ids.append(text_id)
        selected_set.add(text_id)
        print(f"Added: {title}\n")


# ---------------------------------------------------------------------------
# Sensitivity calculations
# ---------------------------------------------------------------------------


def percentile_rank(values: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=values.index, dtype=float)
    mask = values.notna()
    n = int(mask.sum())
    if n == 0:
        return out
    if n == 1:
        out.loc[mask] = 50.0
        return out
    ranks = rankdata(values.loc[mask].to_numpy(dtype=float), method="average")
    out.loc[mask] = 100.0 * (ranks - 1.0) / (n - 1.0)
    return out


def z_scores(values: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=values.index, dtype=float)
    mask = values.notna()
    if int(mask.sum()) < 2:
        return out
    arr = values.loc[mask].astype(float)
    sd = float(arr.std(ddof=0))
    if not math.isfinite(sd) or sd == 0:
        out.loc[mask] = 0.0
        return out
    out.loc[mask] = (arr - float(arr.mean())) / sd
    return out


def _variant_lookup(variants: Sequence[Variant]) -> dict[str, Variant]:
    return {v.variant_id: v for v in variants}


def _same_raw_scale(metric_variants: Sequence[Variant]) -> tuple[bool, str]:
    scales = {v.scale.strip() for v in metric_variants if v.scale.strip()}
    if len(scales) == 1:
        return True, next(iter(scales))
    # Defensive allowance for VAD resources if an export uses cosmetically different
    # labels while all resources are explicitly normalized to 0-1.
    if metric_variants and all(v.lexicon_id in VAD_RESOURCE_IDS for v in metric_variants):
        if scales and all("0" in s and "1" in s and "normal" in s.casefold() for s in scales):
            return True, "normalized 0-1"
    return False, " | ".join(sorted(scales))


def _pair_stats_for_series(values: pd.Series, lookup: dict[str, Variant]) -> dict[str, object]:
    available = values.dropna()
    if len(available) < 2:
        return {
            "min": math.nan,
            "max": math.nan,
            "range": math.nan,
            "average_absolute_change": math.nan,
            "lowest_variant": "",
            "highest_variant": "",
            "largest_pair_change": math.nan,
            "largest_pair_left_variant": "",
            "largest_pair_right_variant": "",
            "largest_pair_change_type": "",
        }
    min_id = str(available.idxmin())
    max_id = str(available.idxmax())
    largest, left, right, change_type = largest_pairwise_difference(available, lookup)
    return {
        "min": float(available.min()),
        "max": float(available.max()),
        "range": float(available.max() - available.min()),
        "average_absolute_change": mean_pairwise_absolute(available.tolist()),
        "lowest_variant": lookup[min_id].label,
        "highest_variant": lookup[max_id].label,
        "largest_pair_change": largest,
        "largest_pair_left_variant": left,
        "largest_pair_right_variant": right,
        "largest_pair_change_type": change_type,
    }


def _within_resource_profile_change(
    corpus_variants: pd.DataFrame,
) -> tuple[float, float]:
    diffs: list[float] = []
    for _, group in corpus_variants.groupby("resource", sort=False):
        vals = group["corpus_mean"].dropna().astype(float).tolist()
        diffs.extend(
            abs(vals[i] - vals[j])
            for i in range(len(vals))
            for j in range(i + 1, len(vals))
        )
    if not diffs:
        return math.nan, math.nan
    return float(max(diffs)), float(np.mean(diffs))


def _same_profile_resource_change(
    corpus_variants: pd.DataFrame,
) -> tuple[float, float]:
    diffs: list[float] = []
    for _, group in corpus_variants.groupby(["analysis_view", "weighting"], sort=False):
        vals = group["corpus_mean"].dropna().astype(float).tolist()
        if group["resource"].nunique() < 2:
            continue
        diffs.extend(
            abs(vals[i] - vals[j])
            for i in range(len(vals))
            for j in range(i + 1, len(vals))
        )
    if not diffs:
        return math.nan, math.nan
    return float(max(diffs)), float(np.mean(diffs))


def analyze_metric(
    metric_key: str,
    metric_label_text: str,
    raw_data: pd.DataFrame,
    variants: Sequence[Variant],
    coverage_threshold: Optional[float],
    sample_mode: str,
    source_work_count: int,
) -> dict[str, object]:
    metric_variants = [v for v in variants if v.metric_key == metric_key]
    variant_ids = [v.variant_id for v in metric_variants]
    lookup = _variant_lookup(metric_variants)
    data = raw_data.loc[raw_data["metric_key"] == metric_key].copy()

    data["passes_value"] = data["value"].notna()
    if coverage_threshold is None:
        data["passes_coverage"] = True
    else:
        data["passes_coverage"] = data["coverage"].notna() & (data["coverage"] >= coverage_threshold)
    data["eligible_for_comparison"] = data["passes_value"] & data["passes_coverage"]

    values = data.pivot(index="text_id", columns="variant_id", values="value").reindex(columns=variant_ids)
    coverage = data.pivot(index="text_id", columns="variant_id", values="coverage").reindex(columns=variant_ids)
    observations = data.pivot(index="text_id", columns="variant_id", values="observations").reindex(columns=variant_ids)
    eligibility = (
        data.pivot(index="text_id", columns="variant_id", values="eligible_for_comparison")
        .reindex(columns=variant_ids)
        .fillna(False)
        .infer_objects()
        .astype(bool)
    )

    metadata_cols = ["text_id", "title", "author", "collection", "date_label", "genre"]
    metadata = data[metadata_cols].drop_duplicates(subset=["text_id"], keep="first").set_index("text_id")

    common_mask = eligibility.all(axis=1)
    common_n = int(common_mask.sum())
    if sample_mode == "common":
        analysis_values = values.loc[common_mask].copy()
        analysis_coverage = coverage.loc[common_mask].copy()
        analysis_observations = observations.loc[common_mask].copy()
    else:
        analysis_values = values.where(eligibility)
        analysis_coverage = coverage.where(eligibility)
        analysis_observations = observations.where(eligibility)

    pct = pd.DataFrame(index=analysis_values.index, columns=analysis_values.columns, dtype=float)
    zdf = pd.DataFrame(index=analysis_values.index, columns=analysis_values.columns, dtype=float)
    for vid in variant_ids:
        pct[vid] = percentile_rank(analysis_values[vid])
        zdf[vid] = z_scores(analysis_values[vid])

    same_scale, scale_label = _same_raw_scale(metric_variants)

    # Corpus means and sample/coverage information by variant.
    corpus_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    for vid in variant_ids:
        variant = lookup[vid]
        eligible_n = int(eligibility[vid].sum())
        variant_values = analysis_values[vid].dropna()
        variant_cov = analysis_coverage[vid].dropna()
        low_coverage_n = int((data.loc[data["variant_id"] == vid, "passes_coverage"] == False).sum())  # noqa: E712
        missing_value_n = int((data.loc[data["variant_id"] == vid, "passes_value"] == False).sum())  # noqa: E712
        corpus_rows.append(
            {
                "metric": metric_label_text,
                "metric_key": metric_key,
                "resource": variant.resource_label,
                "lexicon_id": variant.lexicon_id,
                "profile": variant.profile_label,
                "analysis_view": variant.analysis_view,
                "weighting": variant.weighting,
                "scale": variant.scale,
                "source_work_count": source_work_count,
                "variant_eligible_n": eligible_n,
                "common_qualifying_n": common_n,
                "analysis_n": int(variant_values.notna().sum()),
                "corpus_mean": float(variant_values.mean()) if len(variant_values) else math.nan,
                "corpus_median": float(variant_values.median()) if len(variant_values) else math.nan,
                "corpus_sd_population": float(variant_values.std(ddof=0)) if len(variant_values) else math.nan,
                "corpus_min": float(variant_values.min()) if len(variant_values) else math.nan,
                "corpus_max": float(variant_values.max()) if len(variant_values) else math.nan,
                "mean_coverage": float(variant_cov.mean()) if len(variant_cov) else math.nan,
                "median_coverage": float(variant_cov.median()) if len(variant_cov) else math.nan,
                "sample_mode": sample_mode,
            }
        )
        coverage_rows.append(
            {
                "metric": metric_label_text,
                "resource": variant.resource_label,
                "profile": variant.profile_label,
                "source_work_count": source_work_count,
                "variant_eligible_n": eligible_n,
                "variant_retained_percent": 100.0 * eligible_n / source_work_count if source_work_count else math.nan,
                "common_qualifying_n": common_n,
                "common_retained_percent": 100.0 * common_n / source_work_count if source_work_count else math.nan,
                "analysis_n": int(variant_values.notna().sum()),
                "low_or_missing_coverage_rows": low_coverage_n,
                "missing_value_rows": missing_value_n,
                "mean_coverage": float(variant_cov.mean()) if len(variant_cov) else math.nan,
                "median_coverage": float(variant_cov.median()) if len(variant_cov) else math.nan,
            }
        )
    corpus_variants = pd.DataFrame(corpus_rows)
    coverage_sample = pd.DataFrame(coverage_rows)

    # Pairwise method agreement and mean shifts.
    pair_rows: list[dict[str, object]] = []
    for i, left_id in enumerate(variant_ids):
        for right_id in variant_ids[i + 1 :]:
            left_v = lookup[left_id]
            right_v = lookup[right_id]
            pair = pd.DataFrame(
                {
                    "left": analysis_values[left_id],
                    "right": analysis_values[right_id],
                    "left_pct": pct[left_id],
                    "right_pct": pct[right_id],
                }
            ).dropna(subset=["left", "right"])
            n = len(pair)
            rho = math.nan
            if n >= 2 and pair["left"].nunique() > 1 and pair["right"].nunique() > 1:
                rho = float(spearmanr(pair["left"], pair["right"]).statistic)
            pct_diff = (pair["left_pct"] - pair["right_pct"]).abs()
            raw_comparable = bool(same_scale)
            raw_diff = (pair["left"] - pair["right"]).abs() if raw_comparable else pd.Series(dtype=float)
            worst_poem = ""
            worst_text_id = ""
            if n and len(pct_diff.dropna()):
                worst_text_id = str(pct_diff.idxmax())
                if worst_text_id in metadata.index:
                    worst_poem = str(metadata.loc[worst_text_id].get("title", ""))
            pair_rows.append(
                {
                    "metric": metric_label_text,
                    "comparison_type": comparison_type(left_v, right_v),
                    "left_variant": left_v.label,
                    "right_variant": right_v.label,
                    "left_resource": left_v.resource_label,
                    "right_resource": right_v.resource_label,
                    "left_profile": left_v.profile_label,
                    "right_profile": right_v.profile_label,
                    "n": n,
                    "spearman_rho": rho,
                    "left_corpus_mean": float(pair["left"].mean()) if n else math.nan,
                    "right_corpus_mean": float(pair["right"].mean()) if n else math.nan,
                    "signed_corpus_mean_change_right_minus_left": (
                        float(pair["right"].mean() - pair["left"].mean()) if raw_comparable and n else math.nan
                    ),
                    "absolute_corpus_mean_change": (
                        abs(float(pair["right"].mean() - pair["left"].mean())) if raw_comparable and n else math.nan
                    ),
                    "median_absolute_percentile_difference": float(pct_diff.median()) if n else math.nan,
                    "mean_absolute_percentile_difference": float(pct_diff.mean()) if n else math.nan,
                    "p90_absolute_percentile_difference": float(pct_diff.quantile(0.90)) if n else math.nan,
                    "max_absolute_percentile_difference": float(pct_diff.max()) if n else math.nan,
                    "most_disagreeing_text_id": worst_text_id,
                    "most_disagreeing_poem": worst_poem,
                    "mean_absolute_raw_poem_difference": float(raw_diff.mean()) if raw_comparable and n else math.nan,
                    "max_absolute_raw_poem_difference": float(raw_diff.max()) if raw_comparable and n else math.nan,
                    "raw_scale_comparable": raw_comparable,
                    "scale": scale_label,
                }
            )
    pairwise = pd.DataFrame(pair_rows)

    # Poem-level sensitivity across all selected variants for this metric.
    poem_rows: list[dict[str, object]] = []
    if sample_mode == "common":
        poem_index = analysis_values.index
    else:
        poem_index = analysis_values.index[(analysis_values.notna().sum(axis=1) >= 2)]

    for text_id in poem_index:
        row_values = analysis_values.loc[text_id]
        available = row_values.dropna()
        if len(available) < 2:
            continue
        row_pct = pct.loc[text_id, available.index].dropna()
        if len(row_pct) < 2:
            continue
        md = metadata.loc[text_id] if text_id in metadata.index else pd.Series(dtype=object)
        raw_stats = _pair_stats_for_series(available, lookup) if same_scale else {
            "min": math.nan,
            "max": math.nan,
            "range": math.nan,
            "average_absolute_change": math.nan,
            "lowest_variant": "",
            "highest_variant": "",
            "largest_pair_change": math.nan,
            "largest_pair_left_variant": "",
            "largest_pair_right_variant": "",
            "largest_pair_change_type": "",
        }
        pct_stats = _pair_stats_for_series(row_pct, lookup)
        sort_value = raw_stats["range"] if same_scale and math.isfinite(safe_float(raw_stats["range"])) else pct_stats["range"]
        poem_rows.append(
            {
                "metric": metric_label_text,
                "metric_key": metric_key,
                "text_id": text_id,
                "title": md.get("title", ""),
                "author": md.get("author", ""),
                "collection": md.get("collection", ""),
                "date_label": md.get("date_label", ""),
                "genre": md.get("genre", ""),
                "variants_available": len(available),
                "variants_selected": len(variant_ids),
                "complete_across_selected_variants": len(available) == len(variant_ids),
                "raw_scale_comparable": same_scale,
                "scale": scale_label,
                "raw_min": raw_stats["min"],
                "raw_max": raw_stats["max"],
                "raw_range": raw_stats["range"],
                "average_absolute_raw_change": raw_stats["average_absolute_change"],
                "lowest_raw_variant": raw_stats["lowest_variant"],
                "highest_raw_variant": raw_stats["highest_variant"],
                "largest_raw_change": raw_stats["largest_pair_change"],
                "largest_raw_change_left_variant": raw_stats["largest_pair_left_variant"],
                "largest_raw_change_right_variant": raw_stats["largest_pair_right_variant"],
                "largest_raw_change_type": raw_stats["largest_pair_change_type"],
                "percentile_min": pct_stats["min"],
                "percentile_max": pct_stats["max"],
                "percentile_range": pct_stats["range"],
                "average_absolute_percentile_change": pct_stats["average_absolute_change"],
                "lowest_percentile_variant": pct_stats["lowest_variant"],
                "highest_percentile_variant": pct_stats["highest_variant"],
                "largest_percentile_change": pct_stats["largest_pair_change"],
                "largest_percentile_change_type": pct_stats["largest_pair_change_type"],
                "sensitivity_sort_value": sort_value,
            }
        )
    poem_sensitivity = pd.DataFrame(poem_rows)
    if not poem_sensitivity.empty:
        poem_sensitivity = poem_sensitivity.sort_values(
            ["sensitivity_sort_value", "percentile_range", "title"],
            ascending=[False, False, True],
            kind="stable",
        ).reset_index(drop=True)
        poem_sensitivity.insert(0, "sensitivity_rank", np.arange(1, len(poem_sensitivity) + 1))

    # Long exact evidence table, preserving rows outside common sample so selected
    # poems can still show why they failed a strict common-set stress test.
    long_data = data.copy()
    pct_long = pct.stack(future_stack=True).rename("percentile_rank").reset_index()
    z_long = zdf.stack(future_stack=True).rename("z_score").reset_index()
    long_data = long_data.merge(pct_long, on=["text_id", "variant_id"], how="left")
    long_data = long_data.merge(z_long, on=["text_id", "variant_id"], how="left")
    long_data["in_common_qualifying_set"] = long_data["text_id"].isin(set(values.index[common_mask]))
    if sample_mode == "common":
        long_data["in_analysis_sample"] = long_data["in_common_qualifying_set"]
    else:
        long_data["in_analysis_sample"] = long_data["eligible_for_comparison"]

    # Add corpus mean for the exact variant, useful in poem worksheets.
    mean_map = corpus_variants.set_index(
        corpus_variants.apply(
            lambda r: "|".join(
                (
                    str(r["lexicon_id"]),
                    metric_variants[0].metric if metric_variants else "",
                    metric_variants[0].dimension if metric_variants else "",
                    metric_variants[0].category if metric_variants else "",
                    str(r["analysis_view"]),
                    str(r["weighting"]),
                )
            ),
            axis=1,
        )
    )["corpus_mean"].to_dict()
    long_data["corpus_mean_for_variant"] = long_data["variant_id"].map(mean_map)

    # Wide metric detail for summary workbook.
    wide_index = values.index if sample_mode == "available" else analysis_values.index
    wide = metadata.reindex(wide_index).copy()
    for vid in variant_ids:
        label = lookup[vid].label
        wide[f"RAW | {label}"] = values.reindex(wide_index)[vid]
        wide[f"ELIGIBLE | {label}"] = eligibility.reindex(wide_index)[vid]
        wide[f"PERCENTILE | {label}"] = pct.reindex(wide_index)[vid]
        wide[f"COVERAGE | {label}"] = coverage.reindex(wide_index)[vid]
    wide = wide.reset_index()

    means = corpus_variants.set_index(
        corpus_variants.apply(
            lambda r: next(
                v.variant_id
                for v in metric_variants
                if v.lexicon_id == r["lexicon_id"]
                and v.analysis_view == r["analysis_view"]
                and v.weighting == r["weighting"]
            ),
            axis=1,
        )
    )["corpus_mean"]
    raw_summary = _pair_stats_for_series(means, lookup) if same_scale else {
        "min": math.nan,
        "max": math.nan,
        "range": math.nan,
        "average_absolute_change": math.nan,
        "lowest_variant": "",
        "highest_variant": "",
        "largest_pair_change": math.nan,
        "largest_pair_left_variant": "",
        "largest_pair_right_variant": "",
        "largest_pair_change_type": "",
    }
    within_profile_max, within_profile_avg = _within_resource_profile_change(corpus_variants) if same_scale else (math.nan, math.nan)
    resource_max, resource_avg = _same_profile_resource_change(corpus_variants) if same_scale else (math.nan, math.nan)
    rhos = pairwise["spearman_rho"].dropna() if not pairwise.empty else pd.Series(dtype=float)
    poem_ranges = poem_sensitivity["raw_range"].dropna() if same_scale and not poem_sensitivity.empty else pd.Series(dtype=float)
    pct_ranges = poem_sensitivity["percentile_range"].dropna() if not poem_sensitivity.empty else pd.Series(dtype=float)
    most_sensitive_title = ""
    most_sensitive_change = math.nan
    if not poem_sensitivity.empty:
        most_sensitive_title = str(poem_sensitivity.iloc[0]["title"])
        most_sensitive_change = float(poem_sensitivity.iloc[0]["sensitivity_sort_value"])

    summary = {
        "metric": metric_label_text,
        "metric_key": metric_key,
        "variants_analyzed": len(metric_variants),
        "resources_analyzed": len({v.lexicon_id for v in metric_variants}),
        "profiles_analyzed": len({(v.analysis_view, v.weighting) for v in metric_variants}),
        "source_work_count": source_work_count,
        "common_qualifying_n": common_n,
        "common_retained_percent": 100.0 * common_n / source_work_count if source_work_count else math.nan,
        "sample_mode": sample_mode,
        "raw_scale_comparable_across_variants": same_scale,
        "scale": scale_label,
        "lowest_corpus_mean": raw_summary["min"],
        "highest_corpus_mean": raw_summary["max"],
        "maximum_corpus_mean_change": raw_summary["range"],
        "average_absolute_corpus_mean_change": raw_summary["average_absolute_change"],
        "lowest_corpus_mean_variant": raw_summary["lowest_variant"],
        "highest_corpus_mean_variant": raw_summary["highest_variant"],
        "largest_corpus_change_type": raw_summary["largest_pair_change_type"],
        "maximum_profile_mean_change_within_resource": within_profile_max,
        "average_absolute_profile_mean_change_within_resource": within_profile_avg,
        "maximum_resource_mean_change_same_profile": resource_max,
        "average_absolute_resource_mean_change_same_profile": resource_avg,
        "median_pairwise_spearman_agreement": float(rhos.median()) if len(rhos) else math.nan,
        "minimum_pairwise_spearman_agreement": float(rhos.min()) if len(rhos) else math.nan,
        "median_poem_raw_range": float(poem_ranges.median()) if len(poem_ranges) else math.nan,
        "maximum_poem_raw_range": float(poem_ranges.max()) if len(poem_ranges) else math.nan,
        "median_poem_percentile_range": float(pct_ranges.median()) if len(pct_ranges) else math.nan,
        "maximum_poem_percentile_range": float(pct_ranges.max()) if len(pct_ranges) else math.nan,
        "most_sensitive_poem": most_sensitive_title,
        "most_sensitive_poem_change": most_sensitive_change,
        "mean_change_caution": (
            "Same qualifying poems across variants."
            if sample_mode == "common"
            else "Variant means may use different qualifying poem sets."
        ),
    }

    return {
        "summary": summary,
        "corpus_variants": corpus_variants,
        "coverage_sample": coverage_sample,
        "pairwise": pairwise,
        "poem_sensitivity": poem_sensitivity,
        "long_data": long_data,
        "wide": wide,
    }


def run_sensitivity(
    raw_data: pd.DataFrame,
    concepts: Sequence[MetricConcept],
    variants: Sequence[Variant],
    coverage_threshold: Optional[float],
    sample_mode: str,
    source_work_count: int,
) -> dict[str, object]:
    summaries: list[dict[str, object]] = []
    corpus_frames: list[pd.DataFrame] = []
    coverage_frames: list[pd.DataFrame] = []
    pairwise_frames: list[pd.DataFrame] = []
    poem_frames: list[pd.DataFrame] = []
    long_frames: list[pd.DataFrame] = []
    wides: dict[str, pd.DataFrame] = {}

    for idx, concept in enumerate(concepts, start=1):
        metric_variants = [v for v in variants if v.metric_key == concept.key]
        print(f"[{idx}/{len(concepts)}] {concept.label}: {len(metric_variants)} compatible variant(s)")
        result = analyze_metric(
            concept.key,
            concept.label,
            raw_data,
            variants,
            coverage_threshold,
            sample_mode,
            source_work_count,
        )
        summaries.append(result["summary"])
        corpus_frames.append(result["corpus_variants"])
        coverage_frames.append(result["coverage_sample"])
        pairwise_frames.append(result["pairwise"])
        poem_frames.append(result["poem_sensitivity"])
        long_frames.append(result["long_data"])
        wides[concept.label] = result["wide"]

    summary_df = pd.DataFrame(summaries)
    corpus_df = pd.concat(corpus_frames, ignore_index=True) if corpus_frames else pd.DataFrame()
    coverage_df = pd.concat(coverage_frames, ignore_index=True) if coverage_frames else pd.DataFrame()
    pairwise_df = (
        pd.concat(pairwise_frames, ignore_index=True)
        if any(not frame.empty for frame in pairwise_frames)
        else pd.DataFrame()
    )
    poem_df = (
        pd.concat(poem_frames, ignore_index=True)
        if any(not frame.empty for frame in poem_frames)
        else pd.DataFrame()
    )
    long_df = pd.concat(long_frames, ignore_index=True) if long_frames else pd.DataFrame()
    return {
        "summary": summary_df,
        "corpus": corpus_df,
        "coverage": coverage_df,
        "pairwise": pairwise_df,
        "poems": poem_df,
        "long": long_df,
        "wides": wides,
    }


def build_top_and_stable(poems: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if poems.empty:
        return poems.copy(), poems.copy()
    top = (
        poems.sort_values(["metric", "sensitivity_rank"], kind="stable")
        .groupby("metric", sort=False, group_keys=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    stable = (
        poems.sort_values(
            ["metric", "sensitivity_sort_value", "percentile_range", "title"],
            ascending=[True, True, True, True],
            kind="stable",
        )
        .groupby("metric", sort=False, group_keys=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return top, stable


def build_selected_poem_outputs(
    selected_ids: Sequence[str],
    poem_sensitivity: pd.DataFrame,
    long_data: pd.DataFrame,
    variants: Sequence[Variant],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not selected_ids:
        return pd.DataFrame(), pd.DataFrame()
    selected_set = set(selected_ids)
    summary = poem_sensitivity.loc[poem_sensitivity["text_id"].isin(selected_set)].copy()
    detail = long_data.loc[long_data["text_id"].isin(selected_set)].copy()

    # Add summary rows for selected poems that were not in the strict analysis set.
    metadata = (
        long_data.loc[long_data["text_id"].isin(selected_set), ["text_id", "title", "author"]]
        .drop_duplicates("text_id")
        .set_index("text_id")
    )
    existing = {(str(r.text_id), str(r.metric)) for r in summary.itertuples(index=False)} if not summary.empty else set()
    rows: list[dict[str, object]] = []
    metric_labels = list(dict.fromkeys(v.metric_label for v in variants))
    for text_id in selected_ids:
        for metric in metric_labels:
            if (str(text_id), metric) in existing:
                continue
            subset = detail.loc[(detail["text_id"] == text_id) & (detail["metric_label"] == metric)]
            if subset.empty:
                continue
            eligible = subset.loc[subset["eligible_for_comparison"]]
            md = metadata.loc[text_id] if text_id in metadata.index else pd.Series(dtype=object)
            rows.append(
                {
                    "sensitivity_rank": math.nan,
                    "metric": metric,
                    "metric_key": str(subset.iloc[0]["metric_key"]),
                    "text_id": text_id,
                    "title": md.get("title", ""),
                    "author": md.get("author", ""),
                    "collection": "",
                    "date_label": "",
                    "genre": "",
                    "variants_available": len(eligible),
                    "variants_selected": len(subset),
                    "complete_across_selected_variants": False,
                    "raw_scale_comparable": False,
                    "scale": "",
                    "raw_min": math.nan,
                    "raw_max": math.nan,
                    "raw_range": math.nan,
                    "average_absolute_raw_change": math.nan,
                    "lowest_raw_variant": "",
                    "highest_raw_variant": "",
                    "largest_raw_change": math.nan,
                    "largest_raw_change_left_variant": "",
                    "largest_raw_change_right_variant": "",
                    "largest_raw_change_type": "",
                    "percentile_min": math.nan,
                    "percentile_max": math.nan,
                    "percentile_range": math.nan,
                    "average_absolute_percentile_change": math.nan,
                    "lowest_percentile_variant": "",
                    "highest_percentile_variant": "",
                    "largest_percentile_change": math.nan,
                    "largest_percentile_change_type": "",
                    "sensitivity_sort_value": math.nan,
                }
            )
    if rows:
        summary = pd.concat([summary, pd.DataFrame(rows)], ignore_index=True, sort=False)
    if not summary.empty:
        order = {text_id: idx for idx, text_id in enumerate(selected_ids)}
        summary["_poem_order"] = summary["text_id"].map(order)
        summary = summary.sort_values(["_poem_order", "metric"], kind="stable").drop(columns="_poem_order")
    if not detail.empty:
        order = {text_id: idx for idx, text_id in enumerate(selected_ids)}
        detail["_poem_order"] = detail["text_id"].map(order)
        detail = detail.sort_values(
            ["_poem_order", "metric_label", "resource_label", "analysis_view", "weighting"],
            kind="stable",
        ).drop(columns="_poem_order")
    return summary, detail


# ---------------------------------------------------------------------------
# Simple XLSX writer (standard library only)
# ---------------------------------------------------------------------------


def dataframe_rows(frame: pd.DataFrame, limit: Optional[int] = None) -> list[list[object]]:
    work = frame if limit is None else frame.head(limit)
    rows: list[list[object]] = [list(work.columns)]
    for row in work.itertuples(index=False, name=None):
        rows.append([json_ready(value) for value in row])
    return rows


def _clean_xml_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = "".join(ch for ch in text if ch in "\t\n\r" or ord(ch) >= 32)
    return xml_escape(text, {'"': '&quot;'})


def _excel_col(index: int) -> str:
    letters = ""
    n = index
    while n:
        n, rem = divmod(n - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def _cell_xml(row: int, col: int, value: object, default_header: bool = False) -> str:
    ref = f"{_excel_col(col)}{row}"
    style_id = 1 if default_header else 0
    if isinstance(value, StyledCell):
        style_id = value.style_id
        value = value.value
    style = f' s="{style_id}"' if style_id else ""
    if isinstance(value, Formula):
        expr = _clean_xml_text(value.expression.lstrip("="))
        display = _clean_xml_text(value.display)
        return f'<c r="{ref}" t="str"{style}><f>{expr}</f><v>{display}</v></c>'
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return f'<c r="{ref}"{style}/>'
    if isinstance(value, (bool, np.bool_)):
        return f'<c r="{ref}" t="b"{style}><v>{1 if bool(value) else 0}</v></c>'
    if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value)):
        return f'<c r="{ref}"{style}><v>{float(value):.15g}</v></c>'
    text = _clean_xml_text(value)
    return f'<c r="{ref}" t="inlineStr"{style}><is><t xml:space="preserve">{text}</t></is></c>'


def _worksheet_xml(rows: Sequence[Sequence[object]], freeze_header: bool = True) -> str:
    body: list[str] = []
    for r_idx, row in enumerate(rows, start=1):
        cells = "".join(
            _cell_xml(r_idx, c_idx, value, default_header=(freeze_header and r_idx == 1))
            for c_idx, value in enumerate(row, start=1)
        )
        body.append(f'<row r="{r_idx}">{cells}</row>')
    freeze = (
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
        'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        if freeze_header
        else '<sheetViews><sheetView workbookViewId="0"/></sheetViews>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'{freeze}<sheetData>{"".join(body)}</sheetData></worksheet>'
    )


def _styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="4">'
        '<font><sz val="11"/><name val="Calibri"/><family val="2"/></font>'
        '<font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/><family val="2"/></font>'
        '<font><b/><sz val="14"/><name val="Calibri"/><family val="2"/></font>'
        '<font><b/><sz val="11"/><name val="Calibri"/><family val="2"/></font>'
        '</fonts>'
        '<fills count="4">'
        '<fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FFD9EAF7"/><bgColor indexed="64"/></patternFill></fill>'
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


def write_simple_xlsx(
    path: Path,
    sheets: Sequence[tuple[str, list[list[object]], bool]],
) -> None:
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


def metadata_rows(metadata: dict[str, object]) -> list[list[object]]:
    rows: list[list[object]] = [["field", "value"]]
    for key, value in metadata.items():
        rendered = (
            json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True)
            if isinstance(value, (dict, list, tuple))
            else value
        )
        rows.append([key, rendered])
    return rows


def create_run_folder(output_root: Path, source: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = output_root / f"{slugify(source.stem)}_{stamp}"
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def readme_rows(metadata: dict[str, object]) -> list[list[object]]:
    return [
        [StyledCell("VerseVAD Sensitivity Analysis", 2), ""],
        ["Purpose", "Measure how corpus means and poem-level values change across selected reasonable VerseVAD methodological variants."],
        ["Primary corpus aggregation", "Equal-work mean: each qualifying poem contributes one poem-level metric value."],
        ["Default alignment", "Common qualifying poem set: the same poems are used for every selected variant of a metric."],
        ["Average change", "Average absolute difference across all unique pairs of selected methodological variants."],
        ["VAD scales", "Uses VerseVAD's exported normalized 0-1 VAD values directly. No second normalization is applied."],
        ["Top sensitive poems", "Ranked by raw value range when all selected variants share a scale; otherwise by corpus-percentile range."],
        ["Pairwise Agreement", "Spearman rank agreement asks whether the methodological variants rank poems similarly."],
        ["Coverage", f"Threshold: {metadata.get('coverage_threshold_display', 'None')}"],
        ["Sample alignment", metadata.get("sample_mode_label", "")],
        ["Source", metadata.get("source", "")],
        ["Created", metadata.get("created_at_local", "")],
    ]


def poem_sheet_name(index: int, title: str) -> str:
    prefix = f"P{index:03d} "
    clean = re.sub(r"[\\/*?:\[\]]", "", str(title)).strip() or "Untitled"
    return (prefix + clean)[:31]


def poem_sheet_rows(
    text_id: str,
    title: str,
    author: str,
    poem_summary: pd.DataFrame,
    poem_detail: pd.DataFrame,
) -> list[list[object]]:
    rows: list[list[object]] = [
        [StyledCell(title, 2), ""],
        ["Text ID", text_id],
        ["Author", author],
        [],
        [StyledCell("Sensitivity summary by metric", 3)],
    ]
    summary_cols = [
        "metric",
        "raw_min",
        "raw_max",
        "raw_range",
        "average_absolute_raw_change",
        "percentile_min",
        "percentile_max",
        "percentile_range",
        "average_absolute_percentile_change",
        "largest_raw_change_type",
        "lowest_raw_variant",
        "highest_raw_variant",
        "complete_across_selected_variants",
    ]
    available_cols = [c for c in summary_cols if c in poem_summary.columns]
    if poem_summary.empty:
        rows.append(["No complete sensitivity summary was available for this poem under the selected common-set criteria."])
    else:
        rows.append([StyledCell(c, 1) for c in available_cols])
        for record in poem_summary[available_cols].itertuples(index=False, name=None):
            rows.append([json_ready(v) for v in record])
    rows.extend([[], [StyledCell("Exact methodological variants", 3)]])
    detail_cols = [
        "metric_label",
        "resource_label",
        "profile_label",
        "value",
        "corpus_mean_for_variant",
        "percentile_rank",
        "z_score",
        "coverage",
        "observations",
        "eligible_for_comparison",
        "in_common_qualifying_set",
        "in_analysis_sample",
        "scale",
    ]
    available_detail = [c for c in detail_cols if c in poem_detail.columns]
    rows.append([StyledCell(c, 1) for c in available_detail])
    for record in poem_detail[available_detail].itertuples(index=False, name=None):
        rows.append([json_ready(v) for v in record])
    return rows


def build_poem_profile_workbook(
    path: Path,
    poem_catalog: pd.DataFrame,
    poem_sensitivity: pd.DataFrame,
    long_data: pd.DataFrame,
) -> None:
    catalog = poem_catalog.copy().reset_index(drop=True)
    names: dict[str, str] = {}
    for idx, row in enumerate(catalog.itertuples(index=False), start=1):
        names[str(row.text_id)] = poem_sheet_name(idx, str(row.title))

    index_rows: list[list[object]] = [[
        "Poem #",
        "text_id",
        "title",
        "author",
        "most_sensitive_metric",
        "largest_raw_range",
        "largest_percentile_range",
        "sheet",
        "open",
    ]]
    for idx, row in enumerate(catalog.itertuples(index=False), start=1):
        text_id = str(row.text_id)
        subset = poem_sensitivity.loc[poem_sensitivity["text_id"] == text_id]
        most_metric = ""
        largest_raw = math.nan
        largest_pct = math.nan
        if not subset.empty:
            ordered = subset.sort_values("sensitivity_sort_value", ascending=False, kind="stable")
            most_metric = str(ordered.iloc[0]["metric"])
            raw_vals = subset["raw_range"].dropna()
            pct_vals = subset["percentile_range"].dropna()
            largest_raw = float(raw_vals.max()) if len(raw_vals) else math.nan
            largest_pct = float(pct_vals.max()) if len(pct_vals) else math.nan
        sheet = names[text_id]
        formula = Formula(f'HYPERLINK("#\'{sheet}\'!A1","Open")', "Open")
        index_rows.append([
            idx,
            text_id,
            row.title,
            row.author,
            most_metric,
            largest_raw,
            largest_pct,
            sheet,
            formula,
        ])

    sheets: list[tuple[str, list[list[object]], bool]] = [("Poem Index", index_rows, True)]
    for idx, row in enumerate(catalog.itertuples(index=False), start=1):
        text_id = str(row.text_id)
        summary = poem_sensitivity.loc[poem_sensitivity["text_id"] == text_id].copy()
        detail = long_data.loc[long_data["text_id"] == text_id].copy()
        sheets.append(
            (
                names[text_id],
                poem_sheet_rows(text_id, str(row.title), str(row.author), summary, detail),
                False,
            )
        )
    write_simple_xlsx(path, sheets)


# ---------------------------------------------------------------------------
# Console reporting and export
# ---------------------------------------------------------------------------


def print_results(summary: pd.DataFrame, top_sensitive: pd.DataFrame, top_n: int) -> None:
    print("\n# Sensitivity Results")
    for row in summary.itertuples(index=False):
        print(f"\n{row.metric}")
        print(
            f"  Variants: {int(row.variants_analyzed)} · Resources: {int(row.resources_analyzed)} · "
            f"Profiles: {int(row.profiles_analyzed)}"
        )
        print(
            f"  Common qualifying poems: {int(row.common_qualifying_n)} / {int(row.source_work_count)} "
            f"({fmt_num(row.common_retained_percent, 1)}%)"
        )
        if bool(row.raw_scale_comparable_across_variants):
            print(
                f"  Corpus mean range: {fmt_num(row.lowest_corpus_mean)} to {fmt_num(row.highest_corpus_mean)} "
                f"(maximum change {fmt_num(row.maximum_corpus_mean_change)})"
            )
            print(
                f"  Average absolute corpus-mean change across selected variants: "
                f"{fmt_num(row.average_absolute_corpus_mean_change)}"
            )
            print(f"  Lowest:  {row.lowest_corpus_mean_variant}")
            print(f"  Highest: {row.highest_corpus_mean_variant}")
        else:
            print("  Raw corpus-mean changes are not combined because selected variants do not share one scale.")
        print(
            f"  Pairwise rank agreement: median rho {fmt_num(row.median_pairwise_spearman_agreement)} · "
            f"minimum {fmt_num(row.minimum_pairwise_spearman_agreement)}"
        )
        subset = top_sensitive.loc[top_sensitive["metric"] == row.metric].head(min(top_n, 5))
        if not subset.empty:
            print("  Most method-sensitive poems:")
            for item in subset.itertuples(index=False):
                change = item.raw_range if bool(item.raw_scale_comparable) and math.isfinite(safe_float(item.raw_range)) else item.percentile_range
                unit = "raw units" if bool(item.raw_scale_comparable) else "percentile points"
                print(f"    {item.title} · {fmt_num(change, 3)} {unit}")
            total = len(top_sensitive.loc[top_sensitive["metric"] == row.metric])
            if total > len(subset):
                print(f"    ... full top {total} exported.")


def export_results(
    run_folder: Path,
    results: dict[str, object],
    top_sensitive: pd.DataFrame,
    most_stable: pd.DataFrame,
    selected_summary: pd.DataFrame,
    selected_detail: pd.DataFrame,
    selected_ids: Sequence[str],
    poem_catalog: pd.DataFrame,
    metadata: dict[str, object],
    spec: dict[str, object],
    create_poem_workbook: bool,
) -> None:
    summary: pd.DataFrame = results["summary"]
    corpus: pd.DataFrame = results["corpus"]
    coverage: pd.DataFrame = results["coverage"]
    pairwise: pd.DataFrame = results["pairwise"]
    poems: pd.DataFrame = results["poems"]
    long_data: pd.DataFrame = results["long"]
    wides: dict[str, pd.DataFrame] = results["wides"]

    summary.to_csv(run_folder / "sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    corpus.to_csv(run_folder / "corpus_sensitivity.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(run_folder / "coverage_sample.csv", index=False, encoding="utf-8-sig")
    top_sensitive.to_csv(run_folder / "top_sensitive_poems.csv", index=False, encoding="utf-8-sig")
    most_stable.to_csv(run_folder / "most_stable_poems.csv", index=False, encoding="utf-8-sig")
    selected_summary.to_csv(run_folder / "selected_poem_sensitivity.csv", index=False, encoding="utf-8-sig")
    selected_detail.to_csv(run_folder / "selected_poem_variants.csv", index=False, encoding="utf-8-sig")
    pairwise.to_csv(run_folder / "pairwise_agreement.csv", index=False, encoding="utf-8-sig")
    poems.to_csv(run_folder / "poem_sensitivity.csv", index=False, encoding="utf-8-sig")
    long_data.to_csv(run_folder / "variant_values_long.csv", index=False, encoding="utf-8-sig")
    (run_folder / "analysis_spec.json").write_text(
        json.dumps(json_ready(spec), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    (run_folder / "analysis_metadata.json").write_text(
        json.dumps(json_ready(metadata), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    summary_sheets: list[tuple[str, list[list[object]], bool]] = [
        ("00 Read Me", readme_rows(metadata), False),
        ("01 Corpus Sensitivity", dataframe_rows(summary), True),
        ("02 Corpus Means", dataframe_rows(corpus), True),
        ("03 Top Sensitive Poems", dataframe_rows(top_sensitive), True),
        ("04 Most Stable Poems", dataframe_rows(most_stable), True),
        ("05 Selected Poems", dataframe_rows(selected_summary), True),
        ("06 Coverage & Sample", dataframe_rows(coverage), True),
        ("07 Pairwise Agreement", dataframe_rows(pairwise), True),
    ]
    for idx, (label, frame) in enumerate(wides.items(), start=1):
        summary_sheets.append((f"M{idx:02d} {label}", dataframe_rows(frame), True))

    if selected_ids:
        selected_catalog = poem_catalog.set_index("text_id")
        for idx, text_id in enumerate(selected_ids, start=1):
            if text_id not in selected_catalog.index:
                continue
            row = selected_catalog.loc[text_id]
            psummary = poems.loc[poems["text_id"] == text_id].copy()
            pdetail = long_data.loc[long_data["text_id"] == text_id].copy()
            summary_sheets.append(
                (
                    f"S{idx:02d} {row['title']}",
                    poem_sheet_rows(text_id, str(row["title"]), str(row["author"]), psummary, pdetail),
                    False,
                )
            )

    summary_sheets.append(("Run Metadata", metadata_rows(metadata), True))
    write_simple_xlsx(run_folder / "sensitivity_summary.xlsx", summary_sheets)

    if create_poem_workbook:
        build_poem_profile_workbook(
            run_folder / "sensitivity_poem_profiles.xlsx",
            poem_catalog,
            poems,
            long_data,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive VerseVAD corpus/poem methodological sensitivity analysis."
    )
    parser.add_argument("--version", action="version", version=f"sensitivity.py {__version__}")
    parser.add_argument(
        "--chunksize",
        type=int,
        default=75000,
        help="CSV streaming chunk size (default: 75000).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_console_encoding()
    args = build_parser().parse_args(argv)
    root = project_root()
    source_dir = source_directory(root)
    output_root = export_directory(root)

    print("VerseVAD Sensitivity Analysis")
    print("=============================\n")
    print(
        "Measure how corpus means and poem-level measurements change across selected "
        "VerseVAD scope/weighting profiles and compatible resources."
    )

    source = select_source(discover_sources(source_dir))
    reader = VerseVADCorpusReader(source, chunksize=args.chunksize)

    print("\nValidating selected VerseVAD corpus export...")
    report = reader.validate()
    print(
        f"Validated: {report.work_count:,} works · {report.lexicon_count} lexical resources · "
        f"schema {report.schema_fingerprint}"
    )

    print("\nBuilding sensitivity metric catalog...")
    catalog = reader.catalog().fillna("")
    concepts = build_metric_concepts(catalog)
    print(f"Available metric concepts: {len(concepts)}")

    selected_concepts = choose_metrics(concepts)
    resource_choices = choose_resources_per_metric(catalog, selected_concepts)

    profiles = relevant_profiles(catalog, selected_concepts, resource_choices)
    selected_profiles = choose_profiles(profiles)

    variants = build_variants(catalog, selected_concepts, resource_choices, selected_profiles)
    if not variants:
        raise SystemExit("No compatible metric × resource × profile variants remain after your selections.")

    print("\n# Selected sensitivity space")
    for concept in selected_concepts:
        metric_variants = [v for v in variants if v.metric_key == concept.key]
        print(f"{concept.label}: {len(metric_variants)} compatible variant(s)")
        for v in metric_variants:
            print(f"  - {v.label}")

    print("\nMinimum poem-level coverage threshold")
    print(
        "Each methodological variant must meet this threshold for a poem to qualify.\n"
        "Enter 80 for 80%, or 0.80 as a proportion. Press Enter for no additional filter."
    )
    while True:
        raw = input("Coverage threshold: ")
        try:
            coverage_threshold = parse_coverage_threshold(raw)
            break
        except ValueError as exc:
            print(str(exc))

    print("\nPoem alignment:")
    print("[1] Common qualifying poem set [default, recommended]")
    print("    Uses the SAME qualifying poems for every selected variant of a metric.")
    print("    Best when you want methodological change without sample-composition change as an extra variable.")
    print("[2] Pairwise / available observations")
    print("    Retains more poems, but variant means and pairwise comparisons may use different poem sets.")
    while True:
        raw = input("\nSelection [1]: ").strip()
        if raw in {"", "1"}:
            sample_mode = "common"
            break
        if raw == "2":
            sample_mode = "available"
            break
        print("Enter 1 or 2.")

    print("\nSensitivity output:")
    print("[1] Corpus sensitivity only")
    print("[2] Specific-poem sensitivity only")
    print("[3] Both [default]")
    while True:
        raw = input("Selection [3]: ").strip()
        if raw in {"", "3"}:
            output_mode = "both"
            break
        if raw == "1":
            output_mode = "corpus"
            break
        if raw == "2":
            output_mode = "poems"
            break
        print("Enter 1, 2, or 3.")

    print("\nHow many most-sensitive and most-stable poems should be highlighted per metric?")
    while True:
        raw = input("Top/bottom poems [20]: ").strip()
        if not raw:
            top_n = 20
            break
        try:
            top_n = int(raw)
            if top_n < 1:
                raise ValueError
            break
        except ValueError:
            print("Enter a positive whole number.")

    print("\nLoading selected variants in one streaming pass...")
    raw_data = load_selected_variants(reader, variants)
    print(f"Loaded {len(raw_data):,} exact poem-level metric rows.")
    poem_catalog = poem_catalog_from_data(raw_data)

    selected_ids: list[str] = []
    if output_mode in {"poems", "both"}:
        raw = input("\nTrack specific poems in detail? [y/N]: ").strip().casefold()
        if raw in {"y", "yes"}:
            selected_ids = choose_poems(poem_catalog)

    create_poem_workbook = False
    if output_mode in {"poems", "both"}:
        raw = input(
            "\nCreate exhaustive sensitivity_poem_profiles.xlsx with one worksheet per poem? [Y/n]: "
        ).strip().casefold()
        create_poem_workbook = raw not in {"n", "no"}

    print("\n# Analysis Summary")
    print(f"Corpus source: {source.name}")
    print(f"Metrics selected: {len(selected_concepts)}")
    print(f"Profiles selected: {len(selected_profiles)}")
    print(f"Compatible metric/resource/profile variants: {len(variants)}")
    print(
        "Coverage threshold: "
        + (f"{coverage_threshold * 100:.1f}%" if coverage_threshold is not None else "None")
    )
    print(
        "Sample alignment: "
        + ("common qualifying poem set" if sample_mode == "common" else "pairwise / available observations")
    )
    print(f"Output: {output_mode}")
    print(f"Top/bottom highlighted per metric: {top_n}")
    print(f"Specific poems selected: {len(selected_ids)}")
    print(f"Exhaustive poem workbook: {'Yes' if create_poem_workbook else 'No'}")
    print("\nNo unselected metric, resource, or profile is added automatically.")

    raw = input("\nRun this sensitivity analysis? [Y/n]: ").strip().casefold()
    if raw in {"n", "no"}:
        print("Cancelled.")
        return 0

    print("\nRunning sensitivity calculations...")
    results = run_sensitivity(
        raw_data,
        selected_concepts,
        variants,
        coverage_threshold,
        sample_mode,
        report.work_count,
    )
    top_sensitive, most_stable = build_top_and_stable(results["poems"], top_n)
    selected_summary, selected_detail = build_selected_poem_outputs(
        selected_ids,
        results["poems"],
        results["long"],
        variants,
    )
    print_results(results["summary"], top_sensitive, top_n)

    run_folder = create_run_folder(output_root, source)
    spec = {
        "tool": "sensitivity.py",
        "tool_version": __version__,
        "metrics": [asdict(concept) for concept in selected_concepts],
        "resource_choices": resource_choices,
        "selected_profiles": [
            {"analysis_view": view, "weighting": weighting, "label": profile_label(view, weighting)}
            for view, weighting in selected_profiles
        ],
        "compatible_variants": [asdict(v) for v in variants],
        "coverage_threshold": coverage_threshold,
        "sample_mode": sample_mode,
        "output_mode": output_mode,
        "top_n": top_n,
        "selected_text_ids": selected_ids,
        "create_poem_workbook": create_poem_workbook,
    }
    metadata = {
        "created_at_local": datetime.now().astimezone().isoformat(),
        "source": str(source.resolve()),
        "source_sha256": file_sha256(source),
        "source_kind": report.source_kind,
        "archive_member": report.archive_member,
        "schema_fingerprint": report.schema_fingerprint,
        "work_count_in_source": report.work_count,
        "lexical_resources_in_source": report.lexicon_count,
        "versevad_reader_version": getattr(versevad_reader, "__version__", "unknown"),
        "sensitivity_script_version": __version__,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": __import__("scipy").__version__,
        "coverage_threshold": coverage_threshold,
        "coverage_threshold_display": (
            f"{coverage_threshold * 100:.1f}%" if coverage_threshold is not None else "None"
        ),
        "sample_mode": sample_mode,
        "sample_mode_label": (
            "Common qualifying poem set" if sample_mode == "common" else "Pairwise / available observations"
        ),
        "metric_count": len(selected_concepts),
        "profile_count_selected": len(selected_profiles),
        "compatible_variant_count": len(variants),
        "selected_poem_count": len(selected_ids),
        "exhaustive_poem_workbook": create_poem_workbook,
        "method_note": (
            "Corpus means are equal-work means across poem-level values. In common mode, the same qualifying poems "
            "are used for every selected methodological variant of a metric. Average change is the mean absolute "
            "difference across all unique pairs of selected variants. VerseVAD-exported VAD means are used on their "
            "existing normalized 0-1 scale and are not renormalized by this script."
        ),
    }
    export_results(
        run_folder,
        results,
        top_sensitive,
        most_stable,
        selected_summary,
        selected_detail,
        selected_ids,
        poem_catalog,
        metadata,
        spec,
        create_poem_workbook,
    )

    print("\n# Exports written")
    print(run_folder)
    print("\nResearch-facing files:")
    print("sensitivity_summary.xlsx")
    if create_poem_workbook:
        print("sensitivity_poem_profiles.xlsx  (Poem Index + one worksheet per poem)")
    print("\nAudit/reproducibility files:")
    print("sensitivity_summary.csv")
    print("corpus_sensitivity.csv")
    print("coverage_sample.csv")
    print("top_sensitive_poems.csv")
    print("most_stable_poems.csv")
    print("selected_poem_sensitivity.csv")
    print("selected_poem_variants.csv")
    print("pairwise_agreement.csv")
    print("poem_sensitivity.csv")
    print("variant_values_long.csv")
    print("analysis_spec.json")
    print("analysis_metadata.json")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
