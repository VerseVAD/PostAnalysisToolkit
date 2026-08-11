#!/usr/bin/env python3
"""robustness.py

Interactive leave-one-out robustness analysis for VerseVAD corpus research.

The script is designed to live beside ``versevad_reader.py`` and
``correlation.py`` inside ``versevad_stats/scripts``. It provides two related
robustness analyses:

1. Correlation robustness
   Recompute a selected Spearman correlation after removing each work once.
   Optional Pearson leave-one-out estimates are reported when Pearson is used
   as a sensitivity check. The original correlation retains its bootstrap 95%
   CI and BH-FDR correction; the leave-one-out replicates are influence checks,
   not a new multiple-testing family.

2. Corpus-metric robustness
   Recompute an equal-work corpus statistic after removing each work once.
   The default statistic is the equal-work arithmetic mean, with optional
   median or population SD.

This file also exposes reusable functions used by ``correlation.py`` so the
correlation program can offer an optional leave-one-out robustness step without
maintaining a second implementation of the methodology.

Dependencies
------------
Python 3.10+, numpy, pandas, scipy, and the local VerseVAD scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import zipfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

try:
    import numpy as np
    import pandas as pd
    import scipy
    from scipy.stats import pearsonr, spearmanr
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "robustness.py requires numpy, pandas, and scipy.\n"
        "Install them with:\n  python -m pip install numpy pandas scipy statsmodels"
    ) from exc

from versevad_tools.cli import parse_index_selection
from versevad_tools.core import configure_console_encoding, project_root as shared_project_root

try:
    import versevad_reader
    from versevad_reader import VerseVADCorpusReader, VerseVADReaderError
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "robustness.py could not import versevad_reader.py. Put both files in the same scripts folder."
    ) from exc


__version__ = "0.1.0"
DEFAULT_TOP_INFLUENTIAL = 20

VIEW_LABELS = {
    "all_matched": "All lexical tokens",
    "stopwords_excluded": "Stopword-excluded",
    "content_words": "Content words only",
}
WEIGHT_LABELS = {"token": "Token-weighted", "type": "Type-weighted"}
PROFILE_ORDER = [
    ("all_matched", "token"),
    ("all_matched", "type"),
    ("stopwords_excluded", "token"),
    ("stopwords_excluded", "type"),
    ("content_words", "token"),
    ("content_words", "type"),
]


# ---------------------------------------------------------------------------
# Reusable statistical engine
# ---------------------------------------------------------------------------


def _finite_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _coefficient(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if len(x) < 3 or len(y) < 3:
        return math.nan
    if np.all(x == x[0]) or np.all(y == y[0]):
        return math.nan
    if method == "spearman":
        return float(spearmanr(x, y).statistic)
    if method == "pearson":
        return float(pearsonr(x, y).statistic)
    raise ValueError(f"Unsupported correlation method: {method}")


def _jackknife_summary(estimates: np.ndarray, original: float) -> dict[str, float]:
    valid = estimates[np.isfinite(estimates)]
    if valid.size == 0 or not math.isfinite(original):
        return {
            "loo_mean": math.nan,
            "loo_median": math.nan,
            "loo_min": math.nan,
            "loo_max": math.nan,
            "loo_range": math.nan,
            "max_abs_change": math.nan,
            "jackknife_bias": math.nan,
            "jackknife_se": math.nan,
        }
    mean_loo = float(np.mean(valid))
    n = int(valid.size)
    jackknife_bias = float((n - 1) * (mean_loo - original)) if n > 1 else math.nan
    jackknife_se = (
        float(math.sqrt((n - 1) / n * np.sum((valid - mean_loo) ** 2)))
        if n > 1
        else math.nan
    )
    return {
        "loo_mean": mean_loo,
        "loo_median": float(np.median(valid)),
        "loo_min": float(np.min(valid)),
        "loo_max": float(np.max(valid)),
        "loo_range": float(np.max(valid) - np.min(valid)),
        "max_abs_change": float(np.max(np.abs(valid - original))),
        "jackknife_bias": jackknife_bias,
        "jackknife_se": jackknife_se,
    }


def correlation_leave_one_out(
    paired_used: pd.DataFrame,
    *,
    include_pearson: bool = False,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Leave each analyzed work out once and recompute correlation coefficients.

    ``paired_used`` is the exact analyzed dataset produced by correlation.py,
    containing x_value and y_value plus poem metadata. No additional missing or
    coverage filtering is performed here.
    """

    required = {"x_value", "y_value", "text_id"}
    missing = required.difference(paired_used.columns)
    if missing:
        raise ValueError(f"paired_used is missing required columns: {sorted(missing)}")

    work = paired_used.copy().reset_index(drop=True)
    x = work["x_value"].to_numpy(dtype=float)
    y = work["y_value"].to_numpy(dtype=float)
    n = len(work)
    if n < 4:
        summary = {
            "status": "insufficient_n",
            "n_analyzed": n,
            "original_spearman": _coefficient(x, y, "spearman") if n >= 3 else math.nan,
        }
        return summary, pd.DataFrame()

    original_sp = _coefficient(x, y, "spearman")
    original_pr = _coefficient(x, y, "pearson") if include_pearson else math.nan

    rows: list[dict[str, object]] = []
    sp_values = np.full(n, np.nan, dtype=float)
    pr_values = np.full(n, np.nan, dtype=float)

    for idx in range(n):
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        sp = _coefficient(x[mask], y[mask], "spearman")
        pr = _coefficient(x[mask], y[mask], "pearson") if include_pearson else math.nan
        sp_values[idx] = sp
        pr_values[idx] = pr

        row: dict[str, object] = {
            "removal_number": idx + 1,
            "text_id": work.at[idx, "text_id"],
            "title": work.at[idx, "title"] if "title" in work.columns else "",
            "author": work.at[idx, "author"] if "author" in work.columns else "",
            "x_value_removed": float(x[idx]),
            "y_value_removed": float(y[idx]),
            "original_spearman": original_sp,
            "loo_spearman": sp,
            "delta_spearman": sp - original_sp if math.isfinite(sp) and math.isfinite(original_sp) else math.nan,
            "abs_delta_spearman": abs(sp - original_sp) if math.isfinite(sp) and math.isfinite(original_sp) else math.nan,
        }
        if include_pearson:
            row.update(
                {
                    "original_pearson": original_pr,
                    "loo_pearson": pr,
                    "delta_pearson": pr - original_pr if math.isfinite(pr) and math.isfinite(original_pr) else math.nan,
                    "abs_delta_pearson": abs(pr - original_pr) if math.isfinite(pr) and math.isfinite(original_pr) else math.nan,
                }
            )
        rows.append(row)

    details = pd.DataFrame(rows)
    if not details.empty:
        details["spearman_influence_rank"] = (
            details["abs_delta_spearman"].rank(method="min", ascending=False, na_option="bottom").astype("Int64")
        )
        if include_pearson:
            details["pearson_influence_rank"] = (
                details["abs_delta_pearson"].rank(method="min", ascending=False, na_option="bottom").astype("Int64")
            )

    sp_summary = _jackknife_summary(sp_values, original_sp)
    finite_sp = details.loc[pd.to_numeric(details["loo_spearman"], errors="coerce").notna()].copy()
    if finite_sp.empty:
        most_abs = most_down = most_up = None
    else:
        most_abs = finite_sp.loc[finite_sp["abs_delta_spearman"].astype(float).idxmax()]
        most_down = finite_sp.loc[finite_sp["delta_spearman"].astype(float).idxmin()]
        most_up = finite_sp.loc[finite_sp["delta_spearman"].astype(float).idxmax()]

    if math.isfinite(original_sp) and original_sp > 0:
        sign_reversals = int(np.sum(sp_values <= 0))
    elif math.isfinite(original_sp) and original_sp < 0:
        sign_reversals = int(np.sum(sp_values >= 0))
    else:
        sign_reversals = 0

    summary: dict[str, object] = {
        "status": "ok",
        "n_analyzed": n,
        "leave_one_out_runs": n,
        "original_spearman": original_sp,
        "spearman_loo_mean": sp_summary["loo_mean"],
        "spearman_loo_median": sp_summary["loo_median"],
        "spearman_loo_min": sp_summary["loo_min"],
        "spearman_loo_max": sp_summary["loo_max"],
        "spearman_loo_range": sp_summary["loo_range"],
        "spearman_max_abs_change": sp_summary["max_abs_change"],
        "spearman_jackknife_bias": sp_summary["jackknife_bias"],
        "spearman_jackknife_se": sp_summary["jackknife_se"],
        "spearman_sign_reversals": sign_reversals,
        "spearman_positive_loo_count": int(np.sum(sp_values > 0)),
        "spearman_negative_loo_count": int(np.sum(sp_values < 0)),
        "spearman_zero_loo_count": int(np.sum(sp_values == 0)),
        "most_influential_text_id": "" if most_abs is None else str(most_abs.get("text_id", "")),
        "most_influential_title": "" if most_abs is None else str(most_abs.get("title", "")),
        "most_influential_delta_spearman": math.nan if most_abs is None else float(most_abs["delta_spearman"]),
        "greatest_decrease_title": "" if most_down is None else str(most_down.get("title", "")),
        "greatest_decrease_delta_spearman": math.nan if most_down is None else float(most_down["delta_spearman"]),
        "greatest_increase_title": "" if most_up is None else str(most_up.get("title", "")),
        "greatest_increase_delta_spearman": math.nan if most_up is None else float(most_up["delta_spearman"]),
    }

    if include_pearson:
        pr_summary = _jackknife_summary(pr_values, original_pr)
        finite_pr = details.loc[pd.to_numeric(details["loo_pearson"], errors="coerce").notna()].copy()
        most_pr = None if finite_pr.empty else finite_pr.loc[finite_pr["abs_delta_pearson"].astype(float).idxmax()]
        summary.update(
            {
                "original_pearson": original_pr,
                "pearson_loo_mean": pr_summary["loo_mean"],
                "pearson_loo_median": pr_summary["loo_median"],
                "pearson_loo_min": pr_summary["loo_min"],
                "pearson_loo_max": pr_summary["loo_max"],
                "pearson_loo_range": pr_summary["loo_range"],
                "pearson_max_abs_change": pr_summary["max_abs_change"],
                "pearson_jackknife_bias": pr_summary["jackknife_bias"],
                "pearson_jackknife_se": pr_summary["jackknife_se"],
                "pearson_most_influential_title": "" if most_pr is None else str(most_pr.get("title", "")),
                "pearson_most_influential_delta": math.nan if most_pr is None else float(most_pr["delta_pearson"]),
            }
        )

    return summary, details


def attach_correlation_robustness(
    analyses: Sequence[Any],
    *,
    include_pearson: bool = False,
) -> dict[int, pd.DataFrame]:
    """Attach leave-one-out summaries to correlation.py PairAnalysis results."""

    detail_frames: dict[int, pd.DataFrame] = {}
    for analysis in analyses:
        summary, details = correlation_leave_one_out(
            analysis.paired_used,
            include_pearson=include_pearson,
        )
        for key, value in summary.items():
            analysis.result[f"robustness_{key}"] = value
        detail_frames[int(analysis.pair_number)] = details
    return detail_frames


def _metric_statistic(values: np.ndarray, statistic: str) -> float:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return math.nan
    if statistic == "mean":
        return float(np.mean(vals))
    if statistic == "median":
        return float(np.median(vals))
    if statistic == "population_sd":
        return float(np.std(vals, ddof=0)) if vals.size >= 2 else math.nan
    raise ValueError(f"Unsupported corpus statistic: {statistic}")


def metric_leave_one_out(
    metric_frame: pd.DataFrame,
    *,
    coverage_threshold: float = 0.0,
    statistic: str = "mean",
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Leave each qualifying work out once and recompute an equal-work statistic."""

    frame = metric_frame.copy()
    complete = pd.to_numeric(frame["value"], errors="coerce").notna()
    if coverage_threshold > 0:
        coverage = pd.to_numeric(frame["coverage"], errors="coerce")
        eligible = complete & coverage.notna() & (coverage >= coverage_threshold)
    else:
        eligible = complete
    used = frame.loc[eligible].copy().reset_index(drop=True)
    used["value"] = pd.to_numeric(used["value"], errors="coerce")
    values = used["value"].to_numpy(dtype=float)
    n = len(values)

    original = _metric_statistic(values, statistic)
    if n < 2 or not math.isfinite(original):
        return {
            "status": "insufficient_n",
            "n_analyzed": n,
            "original_estimate": original,
            "statistic": statistic,
        }, pd.DataFrame(), used

    rows: list[dict[str, object]] = []
    loo = np.full(n, np.nan, dtype=float)
    for idx in range(n):
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        estimate = _metric_statistic(values[mask], statistic)
        loo[idx] = estimate
        rows.append(
            {
                "removal_number": idx + 1,
                "text_id": used.at[idx, "text_id"],
                "title": used.at[idx, "title"] if "title" in used.columns else "",
                "author": used.at[idx, "author"] if "author" in used.columns else "",
                "removed_work_value": float(values[idx]),
                "original_estimate": original,
                "loo_estimate": estimate,
                "delta_estimate": estimate - original if math.isfinite(estimate) else math.nan,
                "abs_delta_estimate": abs(estimate - original) if math.isfinite(estimate) else math.nan,
            }
        )
    details = pd.DataFrame(rows)
    details["influence_rank"] = (
        details["abs_delta_estimate"].rank(method="min", ascending=False, na_option="bottom").astype("Int64")
    )

    j = _jackknife_summary(loo, original)
    finite = details.loc[pd.to_numeric(details["loo_estimate"], errors="coerce").notna()].copy()
    most_abs = None if finite.empty else finite.loc[finite["abs_delta_estimate"].astype(float).idxmax()]
    most_down = None if finite.empty else finite.loc[finite["delta_estimate"].astype(float).idxmin()]
    most_up = None if finite.empty else finite.loc[finite["delta_estimate"].astype(float).idxmax()]

    relative = (
        j["max_abs_change"] / abs(original)
        if math.isfinite(j["max_abs_change"]) and original != 0
        else math.nan
    )
    summary = {
        "status": "ok",
        "statistic": statistic,
        "n_analyzed": n,
        "leave_one_out_runs": n,
        "original_estimate": original,
        "loo_mean": j["loo_mean"],
        "loo_median": j["loo_median"],
        "loo_min": j["loo_min"],
        "loo_max": j["loo_max"],
        "loo_range": j["loo_range"],
        "max_abs_change": j["max_abs_change"],
        "max_abs_change_relative_to_original": relative,
        "jackknife_bias": j["jackknife_bias"],
        "jackknife_se": j["jackknife_se"],
        "most_influential_text_id": "" if most_abs is None else str(most_abs.get("text_id", "")),
        "most_influential_title": "" if most_abs is None else str(most_abs.get("title", "")),
        "most_influential_work_value": math.nan if most_abs is None else float(most_abs["removed_work_value"]),
        "most_influential_delta": math.nan if most_abs is None else float(most_abs["delta_estimate"]),
        "greatest_decrease_title": "" if most_down is None else str(most_down.get("title", "")),
        "greatest_decrease_delta": math.nan if most_down is None else float(most_down["delta_estimate"]),
        "greatest_increase_title": "" if most_up is None else str(most_up.get("title", "")),
        "greatest_increase_delta": math.nan if most_up is None else float(most_up["delta_estimate"]),
    }
    return summary, details, used


# ---------------------------------------------------------------------------
# Correlation.py integration exports
# ---------------------------------------------------------------------------


def _slugify(value: str, max_length: int = 70) -> str:
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").casefold()
    return (clean[:max_length].rstrip("_") or "analysis")


def export_correlation_robustness_into_run(
    run_folder: Path,
    analyses: Sequence[Any],
    detail_frames: dict[int, pd.DataFrame],
    *,
    include_pearson: bool,
    xlsx_writer: Any,
) -> None:
    """Write robustness companion files into an existing correlation run folder."""

    robust_dir = run_folder / "robustness"
    robust_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    sheets: list[tuple[str, list[list[object]]]] = []

    for analysis in analyses:
        r = analysis.result
        row = {
            "pair_number": analysis.pair_number,
            "x_metric": analysis.request.x.label,
            "y_metric": analysis.request.y.label,
            "analysis_view": analysis.request.x.analysis_view,
            "weighting": analysis.request.x.weighting,
        }
        row.update({k.removeprefix("robustness_"): v for k, v in r.items() if k.startswith("robustness_")})
        summary_rows.append(row)
        details = detail_frames.get(int(analysis.pair_number), pd.DataFrame())
        filename = (
            f"{int(analysis.pair_number):02d}_"
            f"{_slugify(analysis.request.x.label, 28)}__{_slugify(analysis.request.y.label, 28)}.csv"
        )
        details.to_csv(robust_dir / filename, index=False)
        if not details.empty:
            sheet_name = f"P{int(analysis.pair_number):02d} LOO"
            sheets.append(
                (sheet_name, [details.columns.tolist()] + details.astype(object).where(pd.notna(details), None).values.tolist())
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(run_folder / "robustness_summary.csv", index=False)
    workbook_sheets = [
        (
            "Robustness Summary",
            [summary.columns.tolist()] + summary.astype(object).where(pd.notna(summary), None).values.tolist(),
        )
    ] + sheets
    xlsx_writer(run_folder / "correlation_robustness.xlsx", workbook_sheets)
    metadata = {
        "analysis_type": "leave_one_out_correlation_robustness",
        "created_at": datetime.now().astimezone().isoformat(),
        "robustness_script_version": __version__,
        "include_pearson": include_pearson,
        "method": "remove each analyzed work once and recompute coefficient",
        "fdr_note": "LOO estimates are influence replicates, not a multiple-testing family; no FDR is applied to LOO runs.",
        "bootstrap_note": "The parent correlation run's bootstrap CI applies to the original estimate; LOO runs are not bootstrapped.",
    }
    (run_folder / "robustness_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def print_correlation_robustness_summary(analyses: Sequence[Any]) -> None:
    print("\nLeave-One-Out Robustness")
    print("========================")
    for analysis in analyses:
        r = analysis.result
        status = str(r.get("robustness_status", "not_run"))
        print(f"\n{analysis.pair_number}. {analysis.request.x.label}")
        print(f"   × {analysis.request.y.label}")
        if status != "ok":
            print(f"   Robustness status: {status}")
            continue
        print(f"   Original Spearman rho: {float(r['robustness_original_spearman']):.6f}")
        print(
            f"   Leave-one-out rho range: {float(r['robustness_spearman_loo_min']):.6f} "
            f"to {float(r['robustness_spearman_loo_max']):.6f}"
        )
        print(f"   Median LOO rho: {float(r['robustness_spearman_loo_median']):.6f}")
        print(f"   Largest absolute rho change: {float(r['robustness_spearman_max_abs_change']):.6f}")
        print(f"   Sign reversals: {int(r['robustness_spearman_sign_reversals'])}")
        title = str(r.get("robustness_most_influential_title", "")) or "(untitled work)"
        delta = _finite_float(r.get("robustness_most_influential_delta_spearman"))
        print(f"   Most influential removal: {title}")
        if math.isfinite(delta):
            print(f"   Change when removed: {delta:+.6f}")


# ---------------------------------------------------------------------------
# Standalone CLI helpers
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    return shared_project_root(__file__)


def choose_profiles() -> list[tuple[str, str]]:
    print("\nChoose lexical scope / weighting profiles:")
    for idx, (view, weight) in enumerate(PROFILE_ORDER, start=1):
        print(f"[{idx}] {VIEW_LABELS[view]} · {WEIGHT_LABELS[weight]}")
    print("\nUse A for all, or selections such as 1,3,5 or 5-6.")
    print("Press Enter for Stopword-excluded · Token-weighted.")
    while True:
        raw = input("Selection: ").strip()
        if not raw:
            return [("stopwords_excluded", "token")]
        try:
            idxs = parse_index_selection(raw, len(PROFILE_ORDER))
            return [PROFILE_ORDER[i] for i in idxs]
        except ValueError as exc:
            print(str(exc))


def _map_choice(corr: Any, reader: VerseVADCorpusReader, base: Any, view: str, weight: str) -> Optional[Any]:
    choices = corr.available_metric_choices(reader, view, weight)
    for choice in choices:
        if (
            choice.lexicon_id == base.lexicon_id
            and choice.metric == base.metric
            and choice.dimension == base.dimension
            and choice.category == base.category
        ):
            return choice
    return None


def _select_metrics(corr: Any, choices: Sequence[Any]) -> list[Any]:
    selected: list[Any] = []
    seen: set[tuple[str, str, str, str]] = set()
    while True:
        print("\nAdd corpus metric")
        print("=================")
        metric = corr.select_metric_interactively(choices, "corpus")
        if metric is None:
            if selected:
                break
            print("No metric has been added yet.")
            continue
        key = (metric.lexicon_id, metric.metric, metric.dimension, metric.category)
        if key in seen:
            print("That exact metric/resource is already selected.")
        else:
            selected.append(metric)
            seen.add(key)
            print(f"\nAdded: {metric.label}")
        if not corr.prompt_yes_no("Add another corpus metric?", default=True):
            break
    return selected


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _export_standalone(
    *,
    corr: Any,
    source: Path,
    reader_report: object,
    correlation_summaries: list[dict[str, object]],
    correlation_details: list[tuple[str, pd.DataFrame]],
    metric_summaries: list[dict[str, object]],
    metric_details: list[tuple[str, pd.DataFrame]],
    settings: dict[str, object],
) -> Path:
    root = _project_root()
    output_root = root / "exports" / "robustness"
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone()
    run = output_root / f"{_slugify(source.stem.replace('complete_audit', ''), 45)}_{stamp.strftime('%Y%m%d_%H%M%S')}"
    run.mkdir(parents=True, exist_ok=False)
    detail_dir = run / "influence_details"
    detail_dir.mkdir(parents=True, exist_ok=True)

    corr_summary_df = pd.DataFrame(correlation_summaries)
    metric_summary_df = pd.DataFrame(metric_summaries)
    if not corr_summary_df.empty:
        corr_summary_df.to_csv(run / "correlation_robustness.csv", index=False)
    if not metric_summary_df.empty:
        metric_summary_df.to_csv(run / "metric_robustness.csv", index=False)

    all_influence: list[pd.DataFrame] = []
    workbook_sheets: list[tuple[str, list[list[object]]]] = []

    summary_parts: list[pd.DataFrame] = []
    if not corr_summary_df.empty:
        tmp = corr_summary_df.copy()
        tmp.insert(0, "analysis_kind", "correlation")
        summary_parts.append(tmp)
    if not metric_summary_df.empty:
        tmp = metric_summary_df.copy()
        tmp.insert(0, "analysis_kind", "corpus_metric")
        summary_parts.append(tmp)
    overall = pd.concat(summary_parts, ignore_index=True, sort=False) if summary_parts else pd.DataFrame()
    if not overall.empty:
        overall.to_csv(run / "robustness_summary.csv", index=False)
        workbook_sheets.append(("Robustness Summary", [overall.columns.tolist()] + overall.astype(object).where(pd.notna(overall), None).values.tolist()))

    if not corr_summary_df.empty:
        workbook_sheets.append(("Correlation Robustness", [corr_summary_df.columns.tolist()] + corr_summary_df.astype(object).where(pd.notna(corr_summary_df), None).values.tolist()))
    if not metric_summary_df.empty:
        workbook_sheets.append(("Metric Robustness", [metric_summary_df.columns.tolist()] + metric_summary_df.astype(object).where(pd.notna(metric_summary_df), None).values.tolist()))

    for idx, (label, frame) in enumerate(correlation_details, start=1):
        if frame.empty:
            continue
        out = frame.copy()
        out.insert(0, "analysis", label)
        out.to_csv(detail_dir / f"C{idx:02d}_{_slugify(label, 45)}.csv", index=False)
        all_influence.append(out)
        workbook_sheets.append((f"C{idx:02d} LOO", [out.columns.tolist()] + out.astype(object).where(pd.notna(out), None).values.tolist()))

    for idx, (label, frame) in enumerate(metric_details, start=1):
        if frame.empty:
            continue
        out = frame.copy()
        out.insert(0, "analysis", label)
        out.to_csv(detail_dir / f"M{idx:02d}_{_slugify(label, 45)}.csv", index=False)
        all_influence.append(out)
        workbook_sheets.append((f"M{idx:02d} LOO", [out.columns.tolist()] + out.astype(object).where(pd.notna(out), None).values.tolist()))

    if all_influence:
        pd.concat(all_influence, ignore_index=True, sort=False).to_csv(run / "all_influence_rows.csv", index=False)

    metadata_rows = [["Field", "Value"]] + [[k, json.dumps(v) if isinstance(v, (list, dict)) else v] for k, v in settings.items()]
    workbook_sheets.append(("Run Metadata", metadata_rows))
    corr.write_simple_xlsx(run / "robustness_analysis.xlsx", workbook_sheets)

    report = asdict(reader_report) if hasattr(reader_report, "__dataclass_fields__") else {}
    metadata = {
        "created_at": stamp.isoformat(),
        "source": str(source.resolve()),
        "source_sha256": _file_sha256(source),
        "reader_validation": report,
        "robustness_script_version": __version__,
        "correlation_script_version": getattr(corr, "__version__", "unknown"),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "settings": settings,
        "method_note": "LOO estimates remove exactly one qualifying work and recompute the original statistic.",
        "multiple_testing_note": "BH-FDR applies only to the original selected correlation family within each profile, not to leave-one-out influence replicates.",
    }
    (run / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    (run / "analysis_spec.json").write_text(json.dumps(settings, indent=2, default=str), encoding="utf-8")
    return run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive VerseVAD leave-one-out robustness analysis.")
    parser.add_argument("--source", help="Optional VerseVAD Complete Audit ZIP or corpus_vad_metrics.csv path.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_console_encoding()
    # Lazy import avoids circular imports when correlation.py imports this module
    # only for its statistical engine.
    try:
        import correlation as corr
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("robustness.py requires correlation.py in the same scripts folder.") from exc

    args = _build_parser().parse_args(argv)
    root = _project_root()
    print("VerseVAD Robustness Analysis")
    print("============================\n")

    try:
        if args.source:
            source = Path(args.source).expanduser().resolve()
        else:
            sources = corr.discover_sources(root / "source")
            source = corr.choose_source_interactively(sources)

        reader = VerseVADCorpusReader(source)
        print("Validating selected VerseVAD corpus export...")
        report = reader.validate()
        print(f"Validated: {report.work_count:,} works · {report.lexicon_count} lexical resources · schema {report.schema_fingerprint}")

        mode = corr.prompt_choice(
            "\nWhat would you like to stress-test?",
            [
                ("correlation", "Correlation robustness"),
                ("metric", "Corpus-metric robustness"),
                ("both", "Both"),
            ],
            default=1,
        )
        profiles = choose_profiles()
        reference_view, reference_weight = profiles[0]
        print(f"\nMetric selection uses reference profile: {VIEW_LABELS[reference_view]} · {WEIGHT_LABELS[reference_weight]}")
        reference_choices = corr.available_metric_choices(reader, reference_view, reference_weight)
        if not reference_choices:
            raise VerseVADReaderError("No compatible metrics found for the reference profile.")

        coverage_threshold = corr.prompt_coverage_threshold()
        correlation_pairs: list[Any] = []
        corpus_metrics: list[Any] = []
        bootstrap_resamples = 0
        include_pearson = False
        corpus_statistic = "mean"

        if mode in {"correlation", "both"}:
            correlation_pairs = corr.select_pairs_interactively(reference_choices)
            if not correlation_pairs:
                raise ValueError("No correlation pairs selected.")
            bootstrap_resamples = corr.prompt_bootstrap_resamples()
            include_pearson = corr.prompt_yes_no("Also run Pearson correlations as sensitivity checks?", default=False)

        if mode in {"metric", "both"}:
            corpus_metrics = _select_metrics(corr, reference_choices)
            statistic_choice = corr.prompt_choice(
                "\nCorpus statistic to stress-test:",
                [
                    ("mean", "Equal-work arithmetic mean [default]"),
                    ("median", "Median"),
                    ("population_sd", "Population standard deviation across works"),
                ],
                default=1,
            )
            corpus_statistic = statistic_choice

        print("\nRobustness Analysis Summary")
        print("===========================")
        print(f"Corpus: {source.name}")
        print("Profiles:")
        for view, weight in profiles:
            print(f"  • {VIEW_LABELS[view]} · {WEIGHT_LABELS[weight]}")
        print(f"Coverage threshold: {'none' if coverage_threshold <= 0 else f'{coverage_threshold:.1%}'}")
        if correlation_pairs:
            print(f"Correlation pairs: {len(correlation_pairs)}")
            print(f"Bootstrap: {bootstrap_resamples:,} paired percentile resamples for original correlation 95% CI")
            print("FDR: BH-FDR across the selected original correlation family within each profile")
            print("LOO runs are not separately bootstrapped and are not an FDR family.")
        if corpus_metrics:
            print(f"Corpus metrics: {len(corpus_metrics)}")
            print(f"Corpus statistic: {corpus_statistic}")
        if not corr.prompt_yes_no("\nRun this robustness analysis?", default=True):
            print("Analysis cancelled.")
            return 0

        correlation_summaries: list[dict[str, object]] = []
        correlation_details: list[tuple[str, pd.DataFrame]] = []
        metric_summaries: list[dict[str, object]] = []
        metric_details: list[tuple[str, pd.DataFrame]] = []

        for pidx, (view, weight) in enumerate(profiles, start=1):
            profile_name = f"{VIEW_LABELS[view]} · {WEIGHT_LABELS[weight]}"
            print(f"\nProfile [{pidx}/{len(profiles)}]: {profile_name}")

            if correlation_pairs:
                mapped_pairs: list[Any] = []
                for pair in correlation_pairs:
                    x = _map_choice(corr, reader, pair.x, view, weight)
                    y = _map_choice(corr, reader, pair.y, view, weight)
                    if x is None or y is None:
                        print(f"  Skipping unavailable pair in this profile: {pair.x.label} × {pair.y.label}")
                        continue
                    mapped_pairs.append(corr.PairRequest(x=x, y=y))
                if mapped_pairs:
                    analyses = corr.analyze_pairs(
                        reader,
                        mapped_pairs,
                        coverage_threshold=coverage_threshold,
                        bootstrap_resamples=bootstrap_resamples,
                        include_pearson=include_pearson,
                        include_quadratic=False,
                    )
                    details_by_num = attach_correlation_robustness(analyses, include_pearson=include_pearson)
                    for analysis in analyses:
                        r = analysis.result
                        row: dict[str, object] = {
                            "profile": profile_name,
                            "analysis_view": view,
                            "weighting": weight,
                            "pair_number": analysis.pair_number,
                            "x_metric": analysis.request.x.label,
                            "y_metric": analysis.request.y.label,
                            "n_analyzed": r.get("n_analyzed"),
                            "spearman_rho": r.get("spearman_coefficient"),
                            "spearman_ci95_low": r.get("spearman_ci95_low"),
                            "spearman_ci95_high": r.get("spearman_ci95_high"),
                            "spearman_p_raw": r.get("spearman_p_raw"),
                            "spearman_p_fdr_bh": r.get("spearman_p_fdr_bh"),
                        }
                        row.update({k.removeprefix("robustness_"): v for k, v in r.items() if k.startswith("robustness_")})
                        correlation_summaries.append(row)
                        label = f"{profile_name} | {analysis.request.x.label} × {analysis.request.y.label}"
                        correlation_details.append((label, details_by_num[int(analysis.pair_number)]))
                        print(
                            f"  {analysis.request.x.label} × {analysis.request.y.label}: "
                            f"rho={float(r['spearman_coefficient']):.6f}; "
                            f"LOO range={float(r['robustness_spearman_loo_min']):.6f} to {float(r['robustness_spearman_loo_max']):.6f}; "
                            f"max |Δrho|={float(r['robustness_spearman_max_abs_change']):.6f}"
                        )

            if corpus_metrics:
                for metric in corpus_metrics:
                    mapped = _map_choice(corr, reader, metric, view, weight)
                    if mapped is None:
                        print(f"  Skipping unavailable metric in this profile: {metric.label}")
                        continue
                    frame = reader.select_metric(mapped.to_spec())
                    summary, details, _used = metric_leave_one_out(
                        frame,
                        coverage_threshold=coverage_threshold,
                        statistic=corpus_statistic,
                    )
                    row = {
                        "profile": profile_name,
                        "analysis_view": view,
                        "weighting": weight,
                        "metric": mapped.label,
                        "lexicon_id": mapped.lexicon_id,
                        "metric_id": mapped.metric,
                        "dimension": mapped.dimension,
                        "coverage_threshold": coverage_threshold,
                    }
                    row.update(summary)
                    metric_summaries.append(row)
                    label = f"{profile_name} | {mapped.label} | {corpus_statistic}"
                    metric_details.append((label, details))
                    if summary.get("status") == "ok":
                        print(
                            f"  {mapped.label}: original={float(summary['original_estimate']):.6f}; "
                            f"LOO range={float(summary['loo_min']):.6f} to {float(summary['loo_max']):.6f}; "
                            f"max |Δ|={float(summary['max_abs_change']):.6f}; "
                            f"most influential={summary['most_influential_title']}"
                        )

        settings = {
            "analysis_type": "robustness",
            "mode": mode,
            "profiles": [
                {"analysis_view": view, "weighting": weight, "label": f"{VIEW_LABELS[view]} · {WEIGHT_LABELS[weight]}"}
                for view, weight in profiles
            ],
            "coverage_threshold": coverage_threshold,
            "bootstrap_resamples_original_correlations": bootstrap_resamples if correlation_pairs else None,
            "pearson_sensitivity": include_pearson if correlation_pairs else False,
            "corpus_statistic": corpus_statistic if corpus_metrics else None,
            "correlation_pairs": [
                {"x": pair.x.label, "y": pair.y.label, "x_spec": asdict(pair.x.to_spec()), "y_spec": asdict(pair.y.to_spec())}
                for pair in correlation_pairs
            ],
            "corpus_metrics": [
                {"label": metric.label, "spec": asdict(metric.to_spec())}
                for metric in corpus_metrics
            ],
            "fdr": "Benjamini-Hochberg across original selected correlation pairs within each profile",
            "leave_one_out_fdr": "none; LOO estimates are influence replicates, not separate hypotheses",
        }
        run = _export_standalone(
            corr=corr,
            source=source,
            reader_report=report,
            correlation_summaries=correlation_summaries,
            correlation_details=correlation_details,
            metric_summaries=metric_summaries,
            metric_details=metric_details,
            settings=settings,
        )
        print("\nExports written")
        print("===============")
        print(run)
        print("\nFiles include:")
        print("  robustness_analysis.xlsx")
        print("  robustness_summary.csv")
        if correlation_summaries:
            print("  correlation_robustness.csv")
        if metric_summaries:
            print("  metric_robustness.csv")
        print("  all_influence_rows.csv")
        print("  influence_details/  (one exact leave-one-out table per analysis)")
        print("  analysis_spec.json")
        print("  analysis_metadata.json")
        return 0

    except (VerseVADReaderError, FileNotFoundError, ValueError, zipfile.BadZipFile) as exc:
        print(f"\nVERSEVAD ROBUSTNESS ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nAnalysis cancelled by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
