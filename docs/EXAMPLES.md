# Example Workflows

Examples use generic audit filenames and no copyrighted text.

## Correlation workflow

1. Copy `my_corpus_VerseVAD_complete_audit.zip` into `source/`.
2. Activate the repository's `.venv`.
3. Run `python scripts/correlation.py`.
4. Select one scope/weighting profile.
5. Search for and add explicit metric pairs.
6. Set a coverage threshold.
7. Choose bootstrap and optional sensitivity/robustness settings.
8. Confirm the plan and run.
9. Open `correlation_analysis.xlsx`; use the paired CSVs for independent checks.

## Corpus-comparison workflow

1. Copy 2–5 corpus audits into `source/`.
2. Run `python scripts/compare.py`.
3. Select corpora and confirm human-readable labels.
4. Select constructs, universal resources, and compatible profiles.
5. Set coverage and bootstrap choices.
6. Confirm that like-for-like variants are available everywhere.
7. Review `corpus_results.xlsx` for description and
   `pairwise_differences.xlsx` for differences/effect sizes.

## Single-poem workflow

1. Copy a single-poem Complete Audit ZIP into `source/`.
2. Run `python scripts/single.py`.
3. Select the primary profile and VAD resource.
4. Configure sensitivity, rolling, influence, and contributor options.
5. Open `single_poem_analysis.xlsx` first, then consult the exact CSV evidence.

For a reproducible default smoke run when exactly one compatible single-poem
audit is present:

```text
python scripts/single.py --quick
```

## Audit-reader validation

Validate a corpus audit before a larger run:

```text
python scripts/versevad_reader.py source/my_corpus_VerseVAD_complete_audit.zip
```

Inspect the full metric catalog:

```text
python scripts/versevad_reader.py source/my_corpus_VerseVAD_complete_audit.zip --catalog
```
