#!/usr/bin/env python3
"""correlation.py

Interactive downstream correlation analysis for VerseVAD corpus exports.

This script is designed to live beside ``versevad_reader.py`` inside:

    versevad_stats/
        scripts/
            versevad_reader.py
            correlation.py
        source/
            <VerseVAD Complete Audit ZIPs or corpus_vad_metrics.csv files>
        exports/

Run from the ``versevad_stats`` project folder with:

    python scripts/correlation.py

The normal workflow is fully interactive. The script:

* discovers available VerseVAD corpus sources in ``source/``;
* lets the analyst choose one lexical profile for the run;
* lets the analyst search for and add exactly the metric pairs they want;
* calculates Spearman correlations for every selected pair;
* optionally calculates Pearson correlations as sensitivity checks;
* optionally tests for quadratic (curvilinear) association, including potential
  U-shaped and inverted-U-shaped relationships with an in-range turning point;
* optionally runs leave-one-out robustness analysis for the selected Spearman
  correlations using the shared robustness.py engine;
* applies a user-selected poem-level coverage threshold to BOTH metrics;
* calculates paired percentile-bootstrap 95% confidence intervals;
* applies Benjamini-Hochberg FDR correction across the selected pair family;
* exports a summary CSV, exact paired-data CSVs, an Excel-readable XLSX workbook
  with one worksheet per metric pair, and reproducibility JSON metadata.

The selected metric pairs form the multiple-testing family. This script does
NOT exhaustively correlate every possible VerseVAD metric unless the analyst
explicitly selects those pairs.

Dependencies
------------
Python 3.10+, numpy, pandas, scipy, statsmodels, and versevad_reader.py.
The XLSX writer is implemented with Python's standard library, so no separate
Excel package is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence
from xml.sax.saxutils import escape as xml_escape

try:
    import numpy as np
    import pandas as pd
    import scipy
    import statsmodels
    import statsmodels.api as sm
    from scipy.stats import pearsonr, rankdata, spearmanr, t as student_t
    from statsmodels.stats.multitest import multipletests
except ImportError as exc:  # pragma: no cover - friendly CLI failure
    raise SystemExit(
        "correlation.py requires numpy, pandas, scipy, and statsmodels.\n"
        "Install them with:\n"
        "  python -m pip install numpy pandas scipy statsmodels"
    ) from exc

from versevad_tools.cli import parse_coverage_threshold as shared_parse_coverage_threshold
from versevad_tools.core import configure_console_encoding, project_root as shared_project_root
from versevad_tools.sources import choose_one_source, discover_files

try:
    import versevad_reader
    from versevad_reader import MetricSpec, VerseVADCorpusReader, VerseVADReaderError
except ImportError as exc:  # pragma: no cover - friendly CLI failure
    raise SystemExit(
        "correlation.py could not import versevad_reader.py. Put both files in the "
        "same scripts folder."
    ) from exc


__version__ = "0.3.0"
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 12_345
FDR_ALPHA = 0.05
MIN_QUADRATIC_N = 10
MAX_SEARCH_RESULTS = 20

VIEW_LABELS = {
    "content_words": "Content words only",
    "stopwords_excluded": "Stopword-excluded",
    "all_matched": "All lexical tokens",
}
WEIGHT_LABELS = {"token": "Token-weighted", "type": "Type-weighted"}


@dataclass(frozen=True)
class MetricChoice:
    """A user-facing metric backed by one exact VerseVAD MetricSpec identity."""

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
    def identity_key(self) -> tuple[str, str, str, str, str, str]:
        return (
            self.lexicon_id,
            self.metric,
            self.dimension,
            self.category,
            self.analysis_view,
            self.weighting,
        )


@dataclass(frozen=True)
class PairRequest:
    x: MetricChoice
    y: MetricChoice

    @property
    def unordered_key(self) -> frozenset[tuple[str, str, str, str, str, str]]:
        return frozenset((self.x.identity_key, self.y.identity_key))


@dataclass
class PairAnalysis:
    pair_number: int
    request: PairRequest
    paired_all: pd.DataFrame
    paired_used: pd.DataFrame
    result: dict[str, object]


# ---------------------------------------------------------------------------
# Paths and source discovery
# ---------------------------------------------------------------------------


def project_root() -> Path:
    """Resolve the containing versevad_stats folder from scripts/correlation.py."""

    return shared_project_root(__file__)


def source_directory(root: Path) -> Path:
    return root / "source"


def export_directory(root: Path) -> Path:
    return root / "exports" / "correlation"


def discover_sources(directory: Path) -> list[Path]:
    return discover_files(directory)


def choose_source_interactively(sources: Sequence[Path]) -> Path:
    return choose_one_source(sources)


# ---------------------------------------------------------------------------
# Interactive settings
# ---------------------------------------------------------------------------


def prompt_choice(title: str, options: Sequence[tuple[str, str]], default: int = 1) -> str:
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
    default_hint = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{default_hint}]: ").strip().casefold()
        if raw == "":
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Enter y or n.")


def parse_coverage_threshold(raw: str) -> float:
    value = shared_parse_coverage_threshold(raw, blank=0.0)
    return float(value if value is not None else 0.0)


def prompt_coverage_threshold() -> float:
    print(
        "\nMinimum poem-level coverage threshold\n"
        "A poem is analyzed only when BOTH selected metrics meet this threshold.\n"
        "Enter 75 for 75%, or 0.75 as a proportion. Press Enter for no coverage filter."
    )
    while True:
        raw = input("Coverage threshold: ")
        try:
            return parse_coverage_threshold(raw)
        except ValueError as exc:
            print(exc)


def prompt_bootstrap_resamples() -> int:
    print(
        "\nPaired percentile-bootstrap 95% confidence interval\n"
        f"Press Enter for {DEFAULT_BOOTSTRAP_RESAMPLES:,} resamples."
    )
    while True:
        raw = input("Bootstrap resamples: ").strip()
        if raw == "":
            return DEFAULT_BOOTSTRAP_RESAMPLES
        try:
            value = int(raw.replace(",", ""))
        except ValueError:
            print("Enter a whole number, such as 10000.")
            continue
        if value < 100:
            print("Use at least 100 resamples. 1,000+ is preferable for real analysis.")
            continue
        return value


# ---------------------------------------------------------------------------
# Metric catalog and search
# ---------------------------------------------------------------------------


def _pretty_words(value: str) -> str:
    value = value.replace("_", " ").strip()
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
    if metric.endswith("_mean") or metric == "vad_mean":
        return "mean"
    if metric.endswith("_cumulative"):
        return "cumulative"
    return _pretty_words(metric)


def friendly_metric_label(row: pd.Series) -> str:
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
        clean = dimension.replace("_association", "")
        base = f"{_pretty_words(clean)} association"
    elif lexicon_id == "nrc_emotion_intensity_v1":
        clean = dimension.replace("_intensity", "")
        base = f"{_pretty_words(clean)} intensity"
    else:
        base = _pretty_words(dimension or category or metric)

    # Avoid labels like "Concreteness mean mean" where the raw metric encodes
    # the field name twice.
    label = f"{base} {suffix}".strip()
    return f"{label} · {resource}"


def _metric_priority(choice: MetricChoice) -> tuple[int, str]:
    metric = choice.metric
    ordinary_mean = (
        metric == "vad_mean"
        or metric.endswith("_mean_mean")
        or (metric.endswith("_mean") and metric != "vad_average_deviation_from_poem_mean")
    )
    if ordinary_mean:
        priority = 0
    elif "standard_deviation" in metric:
        priority = 1
    elif metric.endswith("_cumulative"):
        priority = 2
    else:
        priority = 3
    return priority, choice.label.casefold()


def available_metric_choices(
    reader: VerseVADCorpusReader,
    analysis_view: str,
    weighting: str,
) -> list[MetricChoice]:
    catalog = reader.catalog()
    selected = catalog.loc[
        (catalog["analysis_view"] == analysis_view)
        & (catalog["weighting"] == weighting)
        & (~catalog["metric"].isin(["coverage", "type_coverage"]))
    ].copy()

    identity_cols = [
        "lexicon_id",
        "lexicon",
        "metric",
        "dimension",
        "category",
        "scale",
    ]
    selected = selected[identity_cols].drop_duplicates().fillna("")

    choices: list[MetricChoice] = []
    for _, row in selected.iterrows():
        choices.append(
            MetricChoice(
                label=friendly_metric_label(row),
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
    return sorted(choices, key=_metric_priority)


def _normalize_search(value: str) -> str:
    value = value.casefold().replace("acquisition", "aoa acquisition")
    value = re.sub(r"[^a-z0-9.]+", " ", value)
    return " ".join(value.split())


def search_metric_choices(choices: Sequence[MetricChoice], query: str) -> list[MetricChoice]:
    terms = _normalize_search(query).split()
    if not terms:
        return []

    scored: list[tuple[int, tuple[int, str], MetricChoice]] = []
    for choice in choices:
        haystack = _normalize_search(
            " ".join(
                [
                    choice.label,
                    choice.lexicon_id,
                    choice.lexicon,
                    choice.metric,
                    choice.dimension,
                    choice.category,
                    choice.scale,
                ]
            )
        )
        if all(term in haystack for term in terms):
            # Prefer matches in the friendly label, then normal metric priority.
            label_haystack = _normalize_search(choice.label)
            label_hits = sum(term in label_haystack for term in terms)
            scored.append((-label_hits, _metric_priority(choice), choice))

    scored.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in scored]


def select_metric_interactively(
    choices: Sequence[MetricChoice],
    axis_name: str,
) -> Optional[MetricChoice]:
    while True:
        print(
            f"\nChoose {axis_name} metric\n"
            "Search by ordinary terms such as: concreteness, frequency, aoa, "
            "interoceptive, arousal nrc v2, valence warriner.\n"
            "Type Q to cancel this pair."
        )
        query = input("Search: ").strip()
        if query.casefold() in {"q", "quit", "cancel"}:
            return None
        matches = search_metric_choices(choices, query)
        if not matches:
            print("No matching metrics. Try a broader or different search term.")
            continue

        shown = matches[:MAX_SEARCH_RESULTS]
        print(f"\nMatches ({len(matches)} total):")
        for idx, choice in enumerate(shown, start=1):
            print(f"[{idx}] {choice.label}")
            print(
                f"    metric={choice.metric}; dimension={choice.dimension or '(none)'}; "
                f"scale={choice.scale or '(unspecified)'}"
            )
        if len(matches) > len(shown):
            print(
                f"Showing the first {len(shown)} matches. Refine your search if the metric "
                "you want is not shown."
            )

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


def select_pairs_interactively(choices: Sequence[MetricChoice]) -> list[PairRequest]:
    pairs: list[PairRequest] = []
    seen: set[frozenset[tuple[str, str, str, str, str, str]]] = set()

    while True:
        print("\nAdd correlation pair")
        print("====================")
        x = select_metric_interactively(choices, "X")
        if x is None:
            if pairs:
                break
            print("No pair has been added yet.")
            continue
        y = select_metric_interactively(choices, "Y")
        if y is None:
            continue

        request = PairRequest(x=x, y=y)
        if x.identity_key == y.identity_key:
            print("X and Y are the same metric. Choose two different metrics.")
            continue
        if request.unordered_key in seen:
            print("That metric pair is already in this analysis family.")
            continue

        pairs.append(request)
        seen.add(request.unordered_key)
        print(f"\nAdded: {x.label} × {y.label}")
        if not prompt_yes_no("Add another pair?", default=True):
            break

    return pairs


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _point_statistic(x: np.ndarray, y: np.ndarray, method: str) -> tuple[float, float]:
    if method == "spearman":
        result = spearmanr(x, y)
        return float(result.statistic), float(result.pvalue)
    if method == "pearson":
        result = pearsonr(x, y)
        return float(result.statistic), float(result.pvalue)
    raise ValueError(f"Unknown correlation method: {method}")


def _quadratic_not_run_result(n: int = 0, *, status: str = "not_run") -> dict[str, object]:
    return {
        "quadratic_status": status,
        "quadratic_shape": "not_run",
        "quadratic_shape_conditions_met": False,
        "quadratic_endpoint_slopes_significant": False,
        "quadratic_strict_shape_raw": False,
        "quadratic_supported_after_fdr": False,
        "quadratic_p_raw": math.nan,
        "quadratic_p_fdr_bh": math.nan,
        "quadratic_fdr_reject_0_05": False,
        "quadratic_intercept": math.nan,
        "quadratic_linear_term_z": math.nan,
        "quadratic_squared_term_z2": math.nan,
        "quadratic_r_squared": math.nan,
        "quadratic_adjusted_r_squared": math.nan,
        "linear_r_squared": math.nan,
        "quadratic_delta_r_squared": math.nan,
        "quadratic_turning_point_x": math.nan,
        "quadratic_turning_point_within_observed_range": False,
        "quadratic_x_min": math.nan,
        "quadratic_x_max": math.nan,
        "quadratic_x_mean": math.nan,
        "quadratic_x_population_sd": math.nan,
        "quadratic_slope_low_x": math.nan,
        "quadratic_slope_high_x": math.nan,
        "quadratic_slope_low_x_p_one_sided": math.nan,
        "quadratic_slope_high_x_p_one_sided": math.nan,
        "quadratic_model_n": int(n),
    }


def quadratic_curvature_analysis(x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    """Test directional quadratic curvature of Y as a function of X.

    This is a second-order polynomial regression, not a correlation coefficient:

        Y = b0 + b1*z(X) + b2*z(X)^2 + error

    where z(X) is X centered at its sample mean and scaled by its population SD.
    Standardizing X improves numerical stability without changing the fitted curve.

    A *potential U-shape* requires all of the following before multiple-testing
    correction: positive quadratic curvature, a turning point inside the observed
    X range, a negative slope at the low-X endpoint, and a positive slope at the
    high-X endpoint. An inverted U uses the opposite signs. Endpoint slopes are
    tested one-sided in the hypothesized directions. The final supported/not-
    supported label is assigned after BH-FDR correction of the quadratic-term
    p-values across the selected pair family.

    The analysis is directional: X is the predictor/horizontal-axis metric and Y
    is the response/vertical-axis metric.
    """

    result: dict[str, object] = _quadratic_not_run_result(len(x))

    if len(x) != len(y):
        result["quadratic_status"] = "unequal_lengths"
        return result
    if len(x) < MIN_QUADRATIC_N:
        result["quadratic_status"] = f"insufficient_n_lt_{MIN_QUADRATIC_N}"
        return result

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    result["quadratic_model_n"] = int(len(x))
    if len(x) < MIN_QUADRATIC_N:
        result["quadratic_status"] = f"insufficient_finite_n_lt_{MIN_QUADRATIC_N}"
        return result

    x_mean = float(np.mean(x))
    x_sd = float(np.std(x, ddof=0))
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    result.update(
        {
            "quadratic_x_mean": x_mean,
            "quadratic_x_population_sd": x_sd,
            "quadratic_x_min": x_min,
            "quadratic_x_max": x_max,
        }
    )
    if not math.isfinite(x_sd) or x_sd <= 0:
        result["quadratic_status"] = "constant_x"
        return result

    z = (x - x_mean) / x_sd
    z2 = z * z
    design_quadratic = np.column_stack((np.ones(len(z)), z, z2))
    design_linear = np.column_stack((np.ones(len(z)), z))

    try:
        quadratic_model = sm.OLS(y, design_quadratic).fit(cov_type="HC3")
        linear_model = sm.OLS(y, design_linear).fit()
    except Exception:
        result["quadratic_status"] = "model_fit_failed"
        return result

    if len(quadratic_model.params) != 3:
        result["quadratic_status"] = "model_rank_failure"
        return result

    b0, b1, b2 = (float(v) for v in quadratic_model.params)
    p_quad = float(quadratic_model.pvalues[2]) if len(quadratic_model.pvalues) > 2 else math.nan
    linear_r2 = float(linear_model.rsquared)
    quad_r2 = float(quadratic_model.rsquared)
    result.update(
        {
            "quadratic_status": "ok",
            "quadratic_intercept": b0,
            "quadratic_linear_term_z": b1,
            "quadratic_squared_term_z2": b2,
            "quadratic_p_raw": p_quad,
            "linear_r_squared": linear_r2,
            "quadratic_r_squared": quad_r2,
            "quadratic_adjusted_r_squared": float(quadratic_model.rsquared_adj),
            "quadratic_delta_r_squared": quad_r2 - linear_r2,
        }
    )

    if not math.isfinite(b2) or abs(b2) < np.finfo(float).eps:
        result["quadratic_shape"] = "no_quadratic_curvature"
        return result

    turning_z = -b1 / (2.0 * b2)
    turning_x = x_mean + x_sd * turning_z
    turning_inside = bool(math.isfinite(turning_x) and x_min <= turning_x <= x_max)
    result["quadratic_turning_point_x"] = float(turning_x)
    result["quadratic_turning_point_within_observed_range"] = turning_inside

    cov = np.asarray(quadratic_model.cov_params(), dtype=float)
    df_resid = float(quadratic_model.df_resid)

    def endpoint_slope(x_value: float) -> tuple[float, float]:
        z_value = (x_value - x_mean) / x_sd
        # dy/dx = (b1 + 2*b2*z) / SD(X)
        contrast = np.array([0.0, 1.0 / x_sd, 2.0 * z_value / x_sd])
        slope = float(contrast @ np.asarray(quadratic_model.params, dtype=float))
        variance = float(contrast @ cov @ contrast)
        if not math.isfinite(variance) or variance <= 0 or df_resid <= 0:
            return slope, math.nan
        se = math.sqrt(variance)
        t_value = slope / se
        # Return t. The one-sided direction depends on candidate shape and is
        # evaluated below.
        return slope, float(t_value)

    slope_low, t_low = endpoint_slope(x_min)
    slope_high, t_high = endpoint_slope(x_max)
    result["quadratic_slope_low_x"] = slope_low
    result["quadratic_slope_high_x"] = slope_high

    if b2 > 0:
        shape = "U-shaped"
        direction_ok = slope_low < 0 and slope_high > 0
        p_low = float(student_t.cdf(t_low, df_resid)) if math.isfinite(t_low) else math.nan
        p_high = float(student_t.sf(t_high, df_resid)) if math.isfinite(t_high) else math.nan
    else:
        shape = "inverted-U-shaped"
        direction_ok = slope_low > 0 and slope_high < 0
        p_low = float(student_t.sf(t_low, df_resid)) if math.isfinite(t_low) else math.nan
        p_high = float(student_t.cdf(t_high, df_resid)) if math.isfinite(t_high) else math.nan

    endpoint_sig = bool(
        math.isfinite(p_low)
        and math.isfinite(p_high)
        and p_low < 0.05
        and p_high < 0.05
    )
    shape_conditions = bool(turning_inside and direction_ok)
    raw_strict = bool(
        shape_conditions
        and endpoint_sig
        and math.isfinite(p_quad)
        and p_quad < 0.05
    )

    result.update(
        {
            "quadratic_shape": shape,
            "quadratic_shape_conditions_met": shape_conditions,
            "quadratic_endpoint_slopes_significant": endpoint_sig,
            "quadratic_strict_shape_raw": raw_strict,
            "quadratic_slope_low_x_p_one_sided": p_low,
            "quadratic_slope_high_x_p_one_sided": p_high,
        }
    )
    return result


def _rowwise_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_centered = a - np.mean(a, axis=1, keepdims=True)
    b_centered = b - np.mean(b, axis=1, keepdims=True)
    numerator = np.sum(a_centered * b_centered, axis=1)
    denominator = np.sqrt(
        np.sum(a_centered * a_centered, axis=1)
        * np.sum(b_centered * b_centered, axis=1)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        return numerator / denominator


def paired_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str,
    n_resamples: int,
    seed: int,
    confidence_level: float = 0.95,
    batch_size: int = 500,
) -> tuple[float, float, int]:
    """Paired percentile bootstrap CI for Spearman rho or Pearson r.

    X/Y row pairing is retained on every resample. For Spearman, ranks are
    recomputed inside every bootstrap sample, including average ranks for ties.
    Constant bootstrap samples yield NaN and are omitted from percentile bounds.
    """

    if len(x) != len(y):
        raise ValueError("Paired bootstrap requires equal-length X and Y arrays.")
    n = len(x)
    if n < 3:
        return math.nan, math.nan, 0

    rng = np.random.default_rng(seed)
    values: list[np.ndarray] = []
    remaining = int(n_resamples)

    while remaining > 0:
        batch = min(batch_size, remaining)
        indices = rng.integers(0, n, size=(batch, n), endpoint=False)
        xb = x[indices]
        yb = y[indices]
        if method == "spearman":
            xb = rankdata(xb, axis=1, method="average")
            yb = rankdata(yb, axis=1, method="average")
        elif method != "pearson":
            raise ValueError(f"Unknown bootstrap method: {method}")
        values.append(_rowwise_pearson(xb, yb))
        remaining -= batch

    estimates = np.concatenate(values)
    estimates = estimates[np.isfinite(estimates)]
    valid = int(len(estimates))
    if valid == 0:
        return math.nan, math.nan, 0

    alpha = 1.0 - confidence_level
    low, high = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high), valid


def _safe_metric_cache_key(choice: MetricChoice) -> tuple[str, str, str, str, str, str]:
    return choice.identity_key


def _apply_coverage_filter(paired: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, dict[str, int]]:
    complete = paired["x_value"].notna() & paired["y_value"].notna()
    n_total = int(len(paired))
    n_pairwise_complete = int(complete.sum())

    if threshold > 0:
        coverage_ok = (
            paired["x_coverage"].notna()
            & paired["y_coverage"].notna()
            & (paired["x_coverage"] >= threshold)
            & (paired["y_coverage"] >= threshold)
        )
    else:
        coverage_ok = pd.Series(True, index=paired.index)

    used_mask = complete & coverage_ok
    used = paired.loc[used_mask].copy().reset_index(drop=True)
    counts = {
        "n_total_works": n_total,
        "n_pairwise_complete": n_pairwise_complete,
        "n_excluded_missing": n_total - n_pairwise_complete,
        "n_excluded_coverage": int((complete & ~coverage_ok).sum()),
        "n_analyzed": int(used_mask.sum()),
    }
    return used, counts


def analyze_pairs(
    reader: VerseVADCorpusReader,
    pairs: Sequence[PairRequest],
    *,
    coverage_threshold: float,
    bootstrap_resamples: int,
    include_pearson: bool,
    include_quadratic: bool,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> list[PairAnalysis]:
    analyses: list[PairAnalysis] = []
    metric_cache: dict[tuple[str, str, str, str, str, str], pd.DataFrame] = {}

    def get_metric(choice: MetricChoice) -> pd.DataFrame:
        key = _safe_metric_cache_key(choice)
        if key not in metric_cache:
            metric_cache[key] = reader.select_metric(choice.to_spec())
        return metric_cache[key]

    for pair_number, request in enumerate(pairs, start=1):
        print(f"\n[{pair_number}/{len(pairs)}] {request.x.label} × {request.y.label}")

        # Join cached metric selections rather than making pair_metrics rescan a
        # repeated metric every time it appears in multiple pairs.
        x = get_metric(request.x)
        y = get_metric(request.y)
        metadata = [
            "text_id",
            "text_version_id",
            "title",
            "author",
            "collection",
            "date_label",
            "genre",
        ]
        evidence = [
            "value",
            "observations",
            "matched_observations",
            "eligible_observations",
            "coverage",
        ]
        x_keep = x[metadata + evidence].rename(columns={col: f"x_{col}" for col in evidence})
        y_keep = y[metadata + evidence].rename(columns={col: f"y_{col}" for col in evidence})
        paired = x_keep.merge(y_keep, on="text_id", how="outer", suffixes=("_xmeta", "_ymeta"), sort=False)

        for col in metadata[1:]:
            left = f"{col}_xmeta"
            right = f"{col}_ymeta"
            l = paired[left].fillna("").astype(str)
            r = paired[right].fillna("").astype(str)
            conflict = (l != "") & (r != "") & (l != r)
            if conflict.any():
                raise VerseVADReaderError(
                    f"Metadata conflict while pairing {request.x.label} and {request.y.label} "
                    f"for field {col!r}."
                )
            paired[col] = l.where(l != "", r)
            paired = paired.drop(columns=[left, right])

        paired["complete_pair"] = paired["x_value"].notna() & paired["y_value"].notna()
        used, counts = _apply_coverage_filter(paired, coverage_threshold)

        result: dict[str, object] = {
            "pair_number": pair_number,
            "x_metric": request.x.label,
            "y_metric": request.y.label,
            "x_lexicon_id": request.x.lexicon_id,
            "x_metric_id": request.x.metric,
            "x_dimension": request.x.dimension,
            "y_lexicon_id": request.y.lexicon_id,
            "y_metric_id": request.y.metric,
            "y_dimension": request.y.dimension,
            "analysis_view": request.x.analysis_view,
            "weighting": request.x.weighting,
            "coverage_threshold": coverage_threshold,
            **counts,
            "status": "ok",
        }

        if counts["n_analyzed"] < 3:
            result["status"] = "insufficient_n"
            for prefix in ("spearman", "pearson"):
                result[f"{prefix}_coefficient"] = math.nan
                result[f"{prefix}_p_raw"] = math.nan
                result[f"{prefix}_ci95_low"] = math.nan
                result[f"{prefix}_ci95_high"] = math.nan
                result[f"{prefix}_bootstrap_valid_resamples"] = 0
            result.update(_quadratic_not_run_result(counts["n_analyzed"], status="not_applicable" if include_quadratic else "not_run"))
            analyses.append(PairAnalysis(pair_number, request, paired, used, result))
            continue

        xv = used["x_value"].to_numpy(dtype=float)
        yv = used["y_value"].to_numpy(dtype=float)
        x_constant = bool(np.all(xv == xv[0]))
        y_constant = bool(np.all(yv == yv[0]))
        if x_constant or y_constant:
            result["status"] = "constant_x" if x_constant else "constant_y"
            if x_constant and y_constant:
                result["status"] = "constant_x_and_y"
            for prefix in ("spearman", "pearson"):
                result[f"{prefix}_coefficient"] = math.nan
                result[f"{prefix}_p_raw"] = math.nan
                result[f"{prefix}_ci95_low"] = math.nan
                result[f"{prefix}_ci95_high"] = math.nan
                result[f"{prefix}_bootstrap_valid_resamples"] = 0
            result.update(_quadratic_not_run_result(counts["n_analyzed"], status="not_applicable" if include_quadratic else "not_run"))
            analyses.append(PairAnalysis(pair_number, request, paired, used, result))
            continue

        rho, spearman_p = _point_statistic(xv, yv, "spearman")
        sp_low, sp_high, sp_valid = paired_bootstrap_ci(
            xv,
            yv,
            method="spearman",
            n_resamples=bootstrap_resamples,
            seed=bootstrap_seed + pair_number * 100,
        )
        result.update(
            {
                "spearman_coefficient": rho,
                "spearman_p_raw": spearman_p,
                "spearman_ci95_low": sp_low,
                "spearman_ci95_high": sp_high,
                "spearman_bootstrap_valid_resamples": sp_valid,
            }
        )

        if include_pearson:
            pearson_r, pearson_p = _point_statistic(xv, yv, "pearson")
            pr_low, pr_high, pr_valid = paired_bootstrap_ci(
                xv,
                yv,
                method="pearson",
                n_resamples=bootstrap_resamples,
                seed=bootstrap_seed + pair_number * 100 + 1,
            )
            result.update(
                {
                    "pearson_coefficient": pearson_r,
                    "pearson_p_raw": pearson_p,
                    "pearson_ci95_low": pr_low,
                    "pearson_ci95_high": pr_high,
                    "pearson_bootstrap_valid_resamples": pr_valid,
                }
            )
        else:
            result.update(
                {
                    "pearson_coefficient": math.nan,
                    "pearson_p_raw": math.nan,
                    "pearson_ci95_low": math.nan,
                    "pearson_ci95_high": math.nan,
                    "pearson_bootstrap_valid_resamples": 0,
                }
            )

        if include_quadratic:
            result.update(quadratic_curvature_analysis(xv, yv))
        else:
            result.update(_quadratic_not_run_result(counts["n_analyzed"]))

        analyses.append(PairAnalysis(pair_number, request, paired, used, result))

    _apply_fdr(analyses, "spearman")
    if include_pearson:
        _apply_fdr(analyses, "pearson")
    else:
        for analysis in analyses:
            analysis.result["pearson_p_fdr_bh"] = math.nan
            analysis.result["pearson_fdr_reject_0_05"] = False

    if include_quadratic:
        _apply_fdr(analyses, "quadratic")
        for analysis in analyses:
            r = analysis.result
            adjusted = r.get("quadratic_p_fdr_bh", math.nan)
            try:
                adjusted_numeric = float(adjusted)
            except (TypeError, ValueError):
                adjusted_numeric = math.nan
            r["quadratic_supported_after_fdr"] = bool(
                r.get("quadratic_shape_conditions_met", False)
                and r.get("quadratic_endpoint_slopes_significant", False)
                and math.isfinite(adjusted_numeric)
                and adjusted_numeric < FDR_ALPHA
            )
    else:
        for analysis in analyses:
            analysis.result["quadratic_p_fdr_bh"] = math.nan
            analysis.result["quadratic_fdr_reject_0_05"] = False
            analysis.result["quadratic_supported_after_fdr"] = False

    for analysis in analyses:
        r = analysis.result
        r["spearman_p_raw_display"] = format_p(r.get("spearman_p_raw"))
        r["spearman_p_fdr_bh_display"] = format_p(r.get("spearman_p_fdr_bh"))
        r["pearson_p_raw_display"] = format_p(r.get("pearson_p_raw"))
        r["pearson_p_fdr_bh_display"] = format_p(r.get("pearson_p_fdr_bh"))
        r["quadratic_p_raw_display"] = format_p(r.get("quadratic_p_raw"))
        r["quadratic_p_fdr_bh_display"] = format_p(r.get("quadratic_p_fdr_bh"))

    return analyses


def _apply_fdr(analyses: Sequence[PairAnalysis], method: str) -> None:
    p_key = f"{method}_p_raw"
    adjusted_key = f"{method}_p_fdr_bh"
    reject_key = f"{method}_fdr_reject_0_05"

    valid_indices: list[int] = []
    p_values: list[float] = []
    for idx, analysis in enumerate(analyses):
        value = analysis.result.get(p_key, math.nan)
        try:
            numeric = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            numeric = math.nan
        if math.isfinite(numeric):
            valid_indices.append(idx)
            p_values.append(numeric)

    for analysis in analyses:
        analysis.result[adjusted_key] = math.nan
        analysis.result[reject_key] = False

    if not p_values:
        return

    reject, adjusted, _, _ = multipletests(p_values, alpha=FDR_ALPHA, method="fdr_bh")
    for source_idx, adj, rej in zip(valid_indices, adjusted, reject):
        analyses[source_idx].result[adjusted_key] = float(adj)
        analyses[source_idx].result[reject_key] = bool(rej)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def slugify(value: str, *, max_length: int = 70) -> str:
    value = value.casefold()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return (value[:max_length].rstrip("_") or "analysis")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def pair_filename(pair: PairRequest, number: int) -> str:
    x = slugify(pair.x.label.split("·")[0].strip(), max_length=32)
    y = slugify(pair.y.label.split("·")[0].strip(), max_length=32)
    return f"{number:02d}_{x}__{y}.csv"


def pair_sheet_name(pair: PairRequest, number: int, used_names: set[str]) -> str:
    x = pair.x.label.split("·")[0].strip()
    y = pair.y.label.split("·")[0].strip()
    base = re.sub(r"[\\/*?:\[\]]", "", f"{number:02d} {x} x {y}")
    base = re.sub(r"\s+", " ", base).strip()[:31]
    name = base
    suffix = 2
    while name.casefold() in used_names:
        trailer = f" {suffix}"
        name = (base[: 31 - len(trailer)] + trailer).strip()
        suffix += 1
    used_names.add(name.casefold())
    return name


def _pair_plot_frame(analysis: PairAnalysis) -> pd.DataFrame:
    used = analysis.paired_used.copy()
    x_header = analysis.request.x.label
    y_header = analysis.request.y.label
    frame = pd.DataFrame(
        {
            x_header: used["x_value"],
            y_header: used["y_value"],
            "title": used["title"],
            "author": used["author"],
            "collection": used["collection"],
            "date_label": used["date_label"],
            "genre": used["genre"],
            "text_id": used["text_id"],
            "x_coverage": used["x_coverage"],
            "y_coverage": used["y_coverage"],
            "x_observations": used["x_observations"],
            "y_observations": used["y_observations"],
        }
    )
    return frame


def export_analysis(
    analyses: Sequence[PairAnalysis],
    *,
    source: Path,
    reader_report: object,
    coverage_threshold: float,
    bootstrap_resamples: int,
    include_pearson: bool,
    include_quadratic: bool,
    analysis_view: str,
    weighting: str,
    output_root: Path,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Path:
    timestamp = datetime.now().astimezone()
    source_slug = slugify(source.stem.replace("complete_audit", ""), max_length=50)
    run_folder = output_root / f"{source_slug}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
    paired_dir = run_folder / "paired_data"
    run_folder.mkdir(parents=True, exist_ok=False)
    paired_dir.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame([analysis.result for analysis in analyses])
    results_path = run_folder / "correlation_results.csv"
    results.to_csv(results_path, index=False)

    pair_sheet_frames: list[tuple[str, pd.DataFrame]] = []
    used_sheet_names: set[str] = {"results", "run_metadata"}
    for analysis in analyses:
        plot_frame = _pair_plot_frame(analysis)
        plot_frame.to_csv(
            paired_dir / pair_filename(analysis.request, analysis.pair_number),
            index=False,
        )
        sheet_name = pair_sheet_name(analysis.request, analysis.pair_number, used_sheet_names)
        pair_sheet_frames.append((sheet_name, plot_frame))

    spec = {
        "analysis_type": "correlation",
        "analysis_view": analysis_view,
        "weighting": weighting,
        "coverage_threshold": coverage_threshold,
        "bootstrap": {
            "paired": True,
            "confidence_level": 0.95,
            "method": "percentile",
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
        },
        "spearman": True,
        "pearson_sensitivity": include_pearson,
        "quadratic_curvilinear_screen": {
            "enabled": include_quadratic,
            "directional": True,
            "model": "Y ~ X + X^2",
            "x_standardized_for_fit": True,
            "inference": "HC3 heteroskedasticity-robust covariance",
            "support_requires": [
                "BH-FDR-adjusted quadratic-term p < 0.05",
                "turning point inside observed X range",
                "endpoint slopes reverse in the candidate U/inverted-U directions",
                "both directional endpoint-slope tests p < 0.05",
            ],
        },
        "fdr": {
            "method": "Benjamini-Hochberg",
            "alpha": FDR_ALPHA,
            "family_definition": "exactly the selected metric pairs in this run",
            "pearson_family": "separate sensitivity family" if include_pearson else None,
            "quadratic_family": "separate curvature-screen family" if include_quadratic else None,
        },
        "pairs": [
            {
                "pair_number": analysis.pair_number,
                "x_label": analysis.request.x.label,
                "y_label": analysis.request.y.label,
                "x_spec": asdict(analysis.request.x.to_spec()),
                "y_spec": asdict(analysis.request.y.to_spec()),
            }
            for analysis in analyses
        ],
    }
    (run_folder / "analysis_spec.json").write_text(
        json.dumps(_json_ready(spec), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report_dict = asdict(reader_report) if hasattr(reader_report, "__dataclass_fields__") else {}
    metadata = {
        "created_at": timestamp.isoformat(),
        "source_file": str(source.resolve()),
        "source_filename": source.name,
        "source_sha256": file_sha256(source),
        "source_size_bytes": source.stat().st_size,
        "versevad_reader_version": getattr(versevad_reader, "__version__", "unknown"),
        "correlation_script_version": __version__,
        "reader_validation": report_dict,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "statsmodels_version": statsmodels.__version__,
        "statistical_settings": spec,
        "pairwise_missing_handling": "pairwise complete observations; missing metric values excluded",
        "coverage_handling": (
            "no additional coverage filter"
            if coverage_threshold <= 0
            else "both X and Y poem-level coverage must meet or exceed the threshold"
        ),
    }
    metadata_path = run_folder / "analysis_metadata.json"
    metadata_path.write_text(
        json.dumps(_json_ready(metadata), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Excel workbook: Results, Run_Metadata, then one immediately plot-ready sheet
    # per selected pair. Columns A and B on each pair sheet are exactly the X and Y
    # values analyzed, so Excel's Insert > Scatter works without restructuring data.
    metadata_rows = [
        ["Field", "Value"],
        ["Created at", timestamp.isoformat()],
        ["Source filename", source.name],
        ["Source SHA-256", metadata["source_sha256"]],
        ["Analysis profile", f"{VIEW_LABELS[analysis_view]} · {WEIGHT_LABELS[weighting]}"],
        ["Coverage threshold", coverage_threshold],
        ["Bootstrap resamples", bootstrap_resamples],
        ["Bootstrap seed", bootstrap_seed],
        ["Bootstrap CI", "95% paired percentile"],
        ["Primary statistic", "Spearman rho, two-sided"],
        ["Pearson sensitivity", include_pearson],
        ["Quadratic/curvilinear screen", include_quadratic],
        ["Quadratic model", "Directional Y ~ X + X^2; HC3-robust inference" if include_quadratic else "Not run"],
        ["FDR", "Benjamini-Hochberg, alpha=0.05"],
        ["FDR family size", len(analyses)],
        ["Reader version", getattr(versevad_reader, "__version__", "unknown")],
        ["Correlation script version", __version__],
    ]

    workbook_path = run_folder / "correlation_analysis.xlsx"
    workbook_sheets: list[tuple[str, list[list[object]]]] = [
        ("Results", [results.columns.tolist()] + results.astype(object).where(pd.notna(results), None).values.tolist()),
        ("Run_Metadata", metadata_rows),
    ]
    for sheet_name, frame in pair_sheet_frames:
        workbook_sheets.append(
            (
                sheet_name,
                [frame.columns.tolist()]
                + frame.astype(object).where(pd.notna(frame), None).values.tolist(),
            )
        )
    write_simple_xlsx(workbook_path, workbook_sheets)

    return run_folder


# ---------------------------------------------------------------------------
# Minimal standards-compliant XLSX writer (stdlib only)
# ---------------------------------------------------------------------------


def _excel_col(number: int) -> str:
    letters = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def _clean_xml_text(value: object) -> str:
    text = str(value)
    # XML 1.0 legal character range, sufficient for workbook text.
    text = "".join(
        ch
        for ch in text
        if ch in "\t\n\r" or 0x20 <= ord(ch) <= 0xD7FF or 0xE000 <= ord(ch) <= 0xFFFD
    )
    return xml_escape(text, {"\"": "&quot;"})


def _cell_xml(cell_ref: str, value: object, *, header: bool = False) -> str:
    style = ' s="1"' if header else ""
    if value is None:
        return f'<c r="{cell_ref}"{style}/>'
    if isinstance(value, (bool, np.bool_)):
        return f'<c r="{cell_ref}" t="b"{style}><v>{1 if bool(value) else 0}</v></c>'
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric):
            rendered = str(int(numeric)) if numeric.is_integer() else repr(numeric)
            return f'<c r="{cell_ref}"{style}><v>{rendered}</v></c>'
        return f'<c r="{cell_ref}"{style}/>'
    text = _clean_xml_text(value)
    preserve = ' xml:space="preserve"' if text.startswith(" ") or text.endswith(" ") else ""
    return f'<c r="{cell_ref}" t="inlineStr"{style}><is><t{preserve}>{text}</t></is></c>'


def _worksheet_xml(rows: Sequence[Sequence[object]]) -> str:
    row_count = max(1, len(rows))
    col_count = max((len(row) for row in rows), default=1)
    dimension = f"A1:{_excel_col(col_count)}{row_count}"

    # Sensible widths based on visible contents, capped to keep sheets manageable.
    widths: list[float] = []
    for col_idx in range(col_count):
        max_len = 0
        for row in rows[: min(len(rows), 1000)]:
            if col_idx < len(row) and row[col_idx] is not None:
                max_len = max(max_len, len(str(row[col_idx])))
        widths.append(float(min(max(max_len + 2, 10), 36)))

    cols_xml = "".join(
        f'<col min="{idx}" max="{idx}" width="{width:.1f}" customWidth="1"/>'
        for idx, width in enumerate(widths, start=1)
    )

    row_xml_parts: list[str] = []
    for row_number, row in enumerate(rows, start=1):
        cells = "".join(
            _cell_xml(
                f"{_excel_col(col_number)}{row_number}",
                value,
                header=(row_number == 1),
            )
            for col_number, value in enumerate(row, start=1)
        )
        row_xml_parts.append(f'<row r="{row_number}">{cells}</row>')

    auto_filter = f'<autoFilter ref="{dimension}"/>' if rows else ""
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="{dimension}"/>'
        '<sheetViews><sheetView workbookViewId="0">'
        '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
        '</sheetView></sheetViews>'
        '<sheetFormatPr defaultRowHeight="15"/>'
        f'<cols>{cols_xml}</cols>'
        f'<sheetData>{"".join(row_xml_parts)}</sheetData>'
        f'{auto_filter}'
        '</worksheet>'
    )


def _styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2">'
        '<font><sz val="11"/><name val="Calibri"/><family val="2"/></font>'
        '<font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/><family val="2"/></font>'
        '</fonts>'
        '<fills count="3">'
        '<fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill>'
        '</fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="2">'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1" applyAlignment="1"><alignment wrapText="1"/></xf>'
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


def write_simple_xlsx(path: Path, sheets: Sequence[tuple[str, list[list[object]]]]) -> None:
    """Write a simple multi-sheet XLSX without third-party Excel dependencies."""

    if not sheets:
        raise ValueError("At least one worksheet is required.")

    used: set[str] = set()
    normalized: list[tuple[str, list[list[object]]]] = []
    for name, rows in sheets:
        normalized.append((_sanitize_sheet_name(name, used), rows))

    content_overrides = "".join(
        f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for idx in range(1, len(normalized) + 1)
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        f'{content_overrides}'
        '</Types>'
    )

    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '</Relationships>'
    )

    sheets_xml = "".join(
        f'<sheet name="{_clean_xml_text(name)}" sheetId="{idx}" r:id="rId{idx}"/>'
        for idx, (name, _) in enumerate(normalized, start=1)
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{sheets_xml}</sheets>'
        '</workbook>'
    )

    workbook_rels = "".join(
        f'<Relationship Id="rId{idx}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        f'Target="worksheets/sheet{idx}.xml"/>'
        for idx in range(1, len(normalized) + 1)
    )
    workbook_rels += (
        f'<Relationship Id="rId{len(normalized) + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'{workbook_rels}'
        '</Relationships>'
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        archive.writestr("xl/styles.xml", _styles_xml())
        for idx, (_, rows) in enumerate(normalized, start=1):
            archive.writestr(f"xl/worksheets/sheet{idx}.xml", _worksheet_xml(rows))


# ---------------------------------------------------------------------------
# Display and CLI
# ---------------------------------------------------------------------------


def format_p(value: object) -> str:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(numeric):
        return "N/A"
    if numeric < 0.001:
        return "< 0.001"
    return f"{numeric:.3f}"


def p_clause(label: str, value: object) -> str:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return f"{label} = N/A"
    if not math.isfinite(numeric):
        return f"{label} = N/A"
    if numeric < 0.001:
        return f"{label} < 0.001"
    return f"{label} = {numeric:.3f}"


def print_analysis_plan(
    source: Path,
    pairs: Sequence[PairRequest],
    *,
    analysis_view: str,
    weighting: str,
    coverage_threshold: float,
    bootstrap_resamples: int,
    include_pearson: bool,
    include_quadratic: bool,
    include_robustness: bool,
) -> None:
    print("\nAnalysis Summary")
    print("================")
    print(f"Corpus source: {source.name}")
    print(f"Profile: {VIEW_LABELS[analysis_view]} · {WEIGHT_LABELS[weighting]}")
    if coverage_threshold > 0:
        print(f"Coverage threshold: {coverage_threshold:.1%} for BOTH metrics")
    else:
        print("Coverage threshold: none")
    print(f"Bootstrap: {bootstrap_resamples:,} paired percentile resamples, 95% CI")
    print("Primary analysis: Spearman rho, two-sided")
    print(f"Pearson sensitivity analysis: {'Yes' if include_pearson else 'No'}")
    print(f"Quadratic/curvilinear screen: {'Yes' if include_quadratic else 'No'}")
    if include_quadratic:
        print("  Directional model: Y ~ X + X², with HC3-robust inference")
    print(f"Leave-one-out robustness: {'Yes' if include_robustness else 'No'}")
    if include_robustness:
        print("  Each analyzed work is removed once and Spearman rho is recomputed.")
        print("  LOO replicates are influence checks, not a separate FDR family.")
    print(f"BH-FDR family size: {len(pairs)} selected pair(s)")
    print("\nSelected pairs:")
    for idx, pair in enumerate(pairs, start=1):
        print(f"  {idx}. {pair.x.label}")
        print(f"     × {pair.y.label}")


def print_result_summary(analyses: Sequence[PairAnalysis], include_pearson: bool, include_quadratic: bool) -> None:
    print("\nCorrelation Results")
    print("===================")
    for analysis in analyses:
        r = analysis.result
        print(f"\n{analysis.pair_number}. {analysis.request.x.label}")
        print(f"   × {analysis.request.y.label}")
        print(f"   N analyzed: {r['n_analyzed']}")
        if r["status"] != "ok":
            print(f"   Status: {r['status']}")
            continue
        print(
            "   Spearman rho: "
            f"{float(r['spearman_coefficient']):.6f} "
            f"(95% CI {float(r['spearman_ci95_low']):.6f} to "
            f"{float(r['spearman_ci95_high']):.6f})"
        )
        print(
            "   "
            + p_clause("p", r["spearman_p_raw"])
            + "; "
            + p_clause("BH-FDR adjusted p", r["spearman_p_fdr_bh"])
        )
        if include_pearson:
            print(
                "   Pearson r:  "
                f"{float(r['pearson_coefficient']):.6f} "
                f"(95% CI {float(r['pearson_ci95_low']):.6f} to "
                f"{float(r['pearson_ci95_high']):.6f})"
            )
            print(
                "   "
                + p_clause("p", r["pearson_p_raw"])
                + "; "
                + p_clause("BH-FDR adjusted p", r["pearson_p_fdr_bh"])
            )

        if include_quadratic:
            q_status = str(r.get("quadratic_status", "not_run"))
            if q_status != "ok":
                print(f"   Quadratic/curvilinear screen: unavailable ({q_status})")
            else:
                shape = str(r.get("quadratic_shape", "not identified"))
                supported = bool(r.get("quadratic_supported_after_fdr", False))
                turning_inside = bool(r.get("quadratic_turning_point_within_observed_range", False))
                turning = r.get("quadratic_turning_point_x", math.nan)
                delta_r2 = r.get("quadratic_delta_r_squared", math.nan)
                print(
                    f"   Quadratic/curvilinear screen: {shape}; "
                    f"{'SUPPORTED' if supported else 'not supported'}"
                )
                print(
                    "   "
                    + p_clause("Quadratic-term p", r.get("quadratic_p_raw"))
                    + "; "
                    + p_clause("BH-FDR adjusted p", r.get("quadratic_p_fdr_bh"))
                )
                if isinstance(turning, (int, float, np.integer, np.floating)) and math.isfinite(float(turning)):
                    print(
                        f"   Turning point X = {float(turning):.6f}; "
                        f"within observed X range: {'Yes' if turning_inside else 'No'}"
                    )
                if isinstance(delta_r2, (int, float, np.integer, np.floating)) and math.isfinite(float(delta_r2)):
                    print(f"   Delta R-squared over linear model: {float(delta_r2):.6f}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive VerseVAD corpus correlation analysis."
    )
    parser.add_argument(
        "--source",
        help="Optional source ZIP/CSV path. If omitted, choose interactively from source/.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_console_encoding()
    parser = _build_parser()
    args = parser.parse_args(argv)

    root = project_root()
    sources_dir = source_directory(root)
    output_root = export_directory(root)

    print("VerseVAD Correlation Analysis")
    print("=============================\n")

    try:
        if args.source:
            source = Path(args.source).expanduser().resolve()
        else:
            sources = discover_sources(sources_dir)
            source = choose_source_interactively(sources)

        reader = VerseVADCorpusReader(source)
        print("Validating selected VerseVAD corpus export...")
        report = reader.validate()
        print(
            f"Validated: {report.work_count:,} works · {report.lexicon_count} lexical resources · "
            f"schema {report.schema_fingerprint}\n"
        )

        analysis_view = prompt_choice(
            "Choose lexical scope:",
            [
                ("content_words", "Content words only"),
                ("stopwords_excluded", "Stopword-excluded"),
                ("all_matched", "All lexical tokens"),
            ],
            default=1,
        )
        print()
        weighting = prompt_choice(
            "Choose weighting:",
            [("token", "Token-weighted"), ("type", "Type-weighted")],
            default=1,
        )

        print("\nBuilding available metric catalog for this profile...")
        choices = available_metric_choices(reader, analysis_view, weighting)
        if not choices:
            raise VerseVADReaderError("No correlatable metrics were found for the selected profile.")
        print(f"Available metric identities: {len(choices)}")

        pairs = select_pairs_interactively(choices)
        if not pairs:
            print("No pairs selected. Nothing to analyze.")
            return 0

        coverage_threshold = prompt_coverage_threshold()
        bootstrap_resamples = prompt_bootstrap_resamples()
        include_pearson = prompt_yes_no("Also run Pearson correlations as sensitivity checks?", default=False)
        include_quadratic = prompt_yes_no(
            "Also screen for directional quadratic/curvilinear association "
            "(potential U-shaped or inverted-U-shaped patterns)?",
            default=False,
        )
        if include_quadratic:
            print("  Note: this screen is directional. X is treated as the predictor/horizontal-axis metric and Y as the response.")
        include_robustness = prompt_yes_no(
            "Also run leave-one-out robustness for these correlations?",
            default=False,
        )

        print_analysis_plan(
            source,
            pairs,
            analysis_view=analysis_view,
            weighting=weighting,
            coverage_threshold=coverage_threshold,
            bootstrap_resamples=bootstrap_resamples,
            include_pearson=include_pearson,
            include_quadratic=include_quadratic,
            include_robustness=include_robustness,
        )
        if not prompt_yes_no("\nRun this analysis?", default=True):
            print("Analysis cancelled.")
            return 0

        print("\nRunning selected correlations...")
        analyses = analyze_pairs(
            reader,
            pairs,
            coverage_threshold=coverage_threshold,
            bootstrap_resamples=bootstrap_resamples,
            include_pearson=include_pearson,
            include_quadratic=include_quadratic,
        )
        print_result_summary(analyses, include_pearson, include_quadratic)

        robustness_details = {}
        robustness_module = None
        if include_robustness:
            try:
                import robustness as robustness_module
            except ImportError as exc:
                raise VerseVADReaderError(
                    "Leave-one-out robustness was requested, but robustness.py could not be imported. "
                    "Put robustness.py in the same scripts folder as correlation.py."
                ) from exc
            print("\nRunning leave-one-out robustness...")
            robustness_details = robustness_module.attach_correlation_robustness(
                analyses, include_pearson=include_pearson
            )
            robustness_module.print_correlation_robustness_summary(analyses)

        run_folder = export_analysis(
            analyses,
            source=source,
            reader_report=report,
            coverage_threshold=coverage_threshold,
            bootstrap_resamples=bootstrap_resamples,
            include_pearson=include_pearson,
            include_quadratic=include_quadratic,
            analysis_view=analysis_view,
            weighting=weighting,
            output_root=output_root,
        )

        if include_robustness and robustness_module is not None:
            robustness_module.export_correlation_robustness_into_run(
                run_folder,
                analyses,
                robustness_details,
                include_pearson=include_pearson,
                xlsx_writer=write_simple_xlsx,
            )

        print("\nExports written")
        print("===============")
        print(run_folder)
        print("\nFiles include:")
        print("  correlation_results.csv")
        print("  correlation_analysis.xlsx")
        print("  analysis_spec.json")
        print("  analysis_metadata.json")
        print("  paired_data/  (one exact analyzed dataset per metric pair)")
        if include_robustness:
            print("  robustness_summary.csv")
            print("  correlation_robustness.xlsx")
            print("  robustness/  (one exact leave-one-out table per metric pair)")
            print("  robustness_metadata.json")
        print(
            "\nExcel note: each metric-pair worksheet places the analyzed X values in "
            "column A and Y values in column B. Select A:B and use Insert > Scatter."
        )
        return 0

    except (VerseVADReaderError, FileNotFoundError, ValueError, zipfile.BadZipFile) as exc:
        print(f"\nVERSEVAD CORRELATION ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nAnalysis cancelled by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
