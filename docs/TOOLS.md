# Tool Guide

All tools are interactive command-line programs. Run `python scripts/<name>.py
--help` for switches and `--version` for the engine version recorded in a run.

## `correlation.py` — correlation analysis

**Question:** Which selected VerseVAD measurements vary together across works?

**Input:** A Corpus / Research Project Complete Audit ZIP.

**Choices:** One lexical scope/weighting profile, explicit metric pairs,
coverage threshold, bootstrap count, optional Pearson sensitivity, optional
quadratic screen, and optional LOO robustness.

**Calculates:** Spearman rho, paired fixed-seed bootstrap intervals,
Benjamini–Hochberg FDR across the selected pair family, and requested
sensitivity/robustness diagnostics.

**Exports:** Summary CSV, exact paired-data CSVs, Excel workbook, analysis
specification, metadata, and optional robustness evidence.

**Does not claim:** causation, exhaustive discovery, or that statistical
association is automatically important to a literary argument.

## `robustness.py` — leave-one-out robustness

**Question:** Does a selected result depend unusually on one work?

**Input:** A corpus audit compatible with `versevad_reader.py`.

**Choices:** Correlation and/or corpus-metric mode, profiles, metrics, coverage,
and (for corpus metrics) equal-work mean, median, or population SD.

**Calculates:** The original estimate, one estimate per omitted work, range,
maximum absolute change, sign reversals where relevant, influential works, and
jackknife diagnostics.

**Exports:** Workbook, summaries, exact influence rows, specification, metadata.

**Does not claim:** that an influential work is erroneous or should be removed.

## `sensitivity.py` — methodological sensitivity

**Question:** How much do measurements move when reasonable choices change?

**Input:** A Corpus / Research Project Complete Audit ZIP.

**Choices:** Constructs, resources, scope/weight profiles, coverage threshold,
common-set versus available/pairwise alignment, selected poems, and output depth.

**Calculates:** Equal-work corpus values by exact variant, within-metric ranges
and average pairwise changes, agreement, work-level sensitivity, and stable or
sensitive works. Common-set alignment is the recommended default.

**Exports:** Summary and optional poem-profile workbooks, detailed CSVs,
specification, and metadata.

**Does not claim:** that one valid method is universally correct, or that raw
movement is comparable across unrelated scales.

## `anomaly.py` — anomaly/outlier exploration

**Question:** Which works are unusual in a selected corpus, and in what ways?

**Input:** A Corpus / Research Project Complete Audit ZIP. Lexical modes use the
canonical Master Metrics records; broader
module mode uses `corpus_module_metrics.csv` when present.

**Choices:** Evidence layer, profile, mode, metrics/directions, percentile
thresholds or result count, coverage where applicable, and broad-scan safeguards.

**Calculates:** Ranked tails, directional percentile combinations, or balanced
multi-metric unusualness with explanatory evidence.

**Exports:** Results and full rankings/evidence as CSV, Excel, specification,
and metadata.

**Does not claim:** error, pathology, value, authorship, or statistical
significance. “Anomalous” means corpus-relative unusualness.

## `compare.py` — multi-corpus comparison

**Questions:** What does each corpus look like? How far apart are 2–5 corpora?
Which measured qualities distinguish them most strongly?

**Input:** Two to five mutually compatible corpus audits/tables.

**Choices:** Corpora and labels, metrics, universal resources/profiles,
coverage threshold, and bootstrap count.

**Calculates:** Equal-work descriptive distributions, all unique corpus-pair
raw differences, fixed-seed bootstrap intervals for mean differences, and
Cliff's delta for scale-independent ordering.

**Exports:** Corpus and pairwise workbooks; description, coverage, pairwise,
effect-size, and work-value CSVs; specification and metadata.

**Does not claim:** formal hypothesis-test significance or percentage changes.
It never pools all corpus tokens to give long works extra corpus weight.

## `single.py` — single-poem close-reading package

**Question:** How can a large single-poem Complete Audit become a coherent,
auditable computational close-reading package?

**Input:** A Single Poem Complete Audit ZIP. New schema-v3 and compatible legacy
single-poem audits are supported through the shared reader/adapter layer.

**Choices:** Primary profile and VAD source, sensitivity variants, rolling
window settings, influence analysis, and contributor depth. `--quick` uses
documented recommended defaults.

**Calculates/organizes:** Authoritative whole-poem metrics and coverage;
methodological sensitivity; line/stanza/rolling views; lexical contributors;
line/stanza influence; and supporting readability, sound, form, and module data.

**Exports:** A structured workbook, detailed CSV tables, specification, metadata.

**Does not claim:** that movement is a formal change point or volta, or that
computational evidence replaces reading the poem.

## `versevad_reader.py` — infrastructure and audit inspection

This strict reader validates the pinned corpus metrics schema, tolerates extra
columns unless strict mode is requested, catalogs exact metric identities, and
extracts unambiguous poem-level metric tables. It refuses missing required
columns, ambiguous identities, and duplicate work rows. It performs no
inferential statistics.
