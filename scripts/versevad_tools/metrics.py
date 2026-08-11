"""Shared labels for exact VerseVAD metric identities."""

from __future__ import annotations

from collections.abc import Mapping

from .core import pretty_words

VIEW_LABELS = {
    "all_matched": "All lexical tokens",
    "stopwords_excluded": "Stopword-excluded",
    "content_words": "Content words only",
}
WEIGHTING_LABELS = {"token": "Token-weighted", "type": "Type-weighted"}
RESOURCE_SHORT_NAMES = {
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
VAD_RESOURCE_IDS = {"nrc_vad_v1", "nrc_vad_v2_1", "warriner_vad_2013"}


def resource_label(lexicon_id: str, lexicon: str) -> str:
    return RESOURCE_SHORT_NAMES.get(str(lexicon_id), str(lexicon) or pretty_words(lexicon_id))


def profile_label(view: str, weighting: str) -> str:
    return f"{VIEW_LABELS.get(view, pretty_words(view))} · {WEIGHTING_LABELS.get(weighting, pretty_words(weighting))}"


def statistic_suffix(metric: str) -> str:
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
    if metric == "coverage":
        return "coverage"
    if metric == "type_coverage":
        return "type coverage"
    if metric.endswith("_standard_deviation") or metric == "vad_standard_deviation":
        return "SD"
    if metric.endswith("_mean") or metric == "vad_mean" or metric.endswith("_mean_mean"):
        return "mean"
    if metric.endswith("_cumulative"):
        return "cumulative"
    return pretty_words(metric)


def friendly_metric_label(row: Mapping[str, object]) -> str:
    lexicon_id = str(row.get("lexicon_id", ""))
    metric = str(row.get("metric", ""))
    dimension = str(row.get("dimension", "") or "")
    category = str(row.get("category", "") or "")
    suffix = statistic_suffix(metric)
    if metric == "coverage":
        return "Coverage"
    if metric == "type_coverage":
        return "Type coverage"
    if lexicon_id == "brysbaert-concreteness-2014":
        base = "Concreteness"
    elif lexicon_id == "kuperman-aoa-2012-erratum-supplement":
        base = "Age of Acquisition (AoA)"
    elif lexicon_id == "subtlex-us-zipf-official":
        base = "Frequency (Zipf)"
    elif lexicon_id == "versevad_lexical_style" and dimension == "mean_word_length":
        base = "Mean word length"
    elif lexicon_id == "lancaster-sensorimotor-2020":
        base = f"{pretty_words(dimension)} sensorimotor"
    elif lexicon_id in VAD_RESOURCE_IDS:
        base = pretty_words(dimension)
    elif lexicon_id == "nrc_emotion_v0_92":
        base = f"{pretty_words(dimension.replace('_association', ''))} association"
    elif lexicon_id == "nrc_emotion_intensity_v1":
        base = f"{pretty_words(dimension.replace('_intensity', ''))} intensity"
    else:
        base = pretty_words(dimension or category or metric)
    if suffix.casefold() in base.casefold():
        return base.strip()
    return f"{base} {suffix}".strip()
