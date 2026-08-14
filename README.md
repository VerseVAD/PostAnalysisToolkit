# VerseVAD Post-Analysis Toolkit

The **VerseVAD Post-Analysis Toolkit** is an optional, command-line collection
of transparent research workflows for [VerseVAD](https://github.com/VerseVAD/VerseVAD)
Complete Audit exports. VerseVAD performs text measurement and produces audit
evidence; this repository helps researchers carry out selected downstream
correlation, robustness, sensitivity, anomaly, corpus-comparison, and
single-poem analyses.

This toolkit is not required to use VerseVAD. The exported CSV data remain open
to independent analysis in Python, R, SPSS, Excel, or another environment.

## What this toolkit is—and is not

It is command-line Python software for reproducible research. It is **not**
VerseVAD itself, a GUI, Streamlit application, web server, automated literary
interpretation system, or substitute for statistical judgment and close
reading. Its outputs describe measured evidence; they do not establish
causation, importance, quality, or a definitive interpretation.

## Included tools

| Script | Research purpose | Expected input | Main output area |
|---|---|---|---|
| `correlation.py` | Ask which selected measurements vary together across works | Corpus Complete Audit ZIP | `exports/correlation/` |
| `robustness.py` | Test whether one work disproportionately influences a selected correlation or corpus statistic | Corpus Complete Audit | `exports/robustness/` |
| `sensitivity.py` | Examine how results move across scope, weighting, resource, and sample-alignment choices | Corpus Complete Audit | `exports/sensitivity/` |
| `anomaly.py` | Explore single-metric extremes, directional combinations, or broad unusual profiles | Corpus Complete Audit; broad mode reads retained module evidence through the shared audit adapter | `exports/anomalies/` |
| `compare.py` | Describe and compare 2–5 corpora using like-for-like methods and effect sizes | Two to five corpus Complete Audits | `exports/compare_corpus/` |
| `single.py` | Turn one single-poem audit into a structured computational close-reading package | Single Poem Complete Audit ZIP | `exports/single_poem/` |

`versevad_reader.py` is the strict schema-validation and metric-extraction
infrastructure used by corpus tools. It can also validate or inspect a corpus
audit from the terminal, but it does not perform inferential analysis.

## Input contract

- `single.py` accepts a **Single Poem Complete Audit ZIP**.
- Every other analytical utility accepts a **Corpus / Research Project Complete Audit ZIP**.
- Compare Poems and Current View exports are intentionally rejected as statistical inputs.
- For export schema 3.0, the shared reader uses
  `03_MASTER_DATA/Master_Metrics.csv`; readable reports and focused tables are
  presentation files, not statistical sources.
- Legacy Single Poem and Corpus Complete Audits remain supported through shared
  compatibility adapters where their required evidence is present.

See [Tool Guide](docs/TOOLS.md) for the questions, choices, calculations,
outputs, and interpretive cautions for each script.

## Requirements

- Python 3.10 through 3.14
- Windows, macOS, or Linux
- No browser, GUI, or web server

The dependency set is deliberately small: NumPy, pandas, SciPy, statsmodels,
and openpyxl. The exact compatible ranges are declared in `pyproject.toml`.

## Installation

Clone this repository:

```text
git clone https://github.com/VerseVAD/PostAnalysisToolkit.git
cd PostAnalysisToolkit
```

### Windows PowerShell

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts\correlation.py --version
```

If `python` is not found, install a current supported Python release from
[python.org](https://www.python.org/downloads/) and reopen PowerShell.

### macOS or Linux Terminal

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/correlation.py --version
```

On macOS, install a current supported Python release from
[python.org](https://www.python.org/downloads/macos/) if `python3` is missing or
too old. Linux package names vary by distribution.

For a step-by-step explanation of terminals, environments, and activation, see
[Getting Started](docs/GETTING_STARTED.md).

## Folder structure

```text
PostAnalysisToolkit/
├── scripts/                 analysis programs and strict audit reader
│   └── versevad_tools/      shared CLI, path, source, and labeling helpers
├── source/                  user-supplied VerseVAD audits (Git-ignored)
├── exports/                 generated analysis runs (Git-ignored)
├── docs/                    user, tool, example, and methodology guides
├── tests/                   focused regression tests
├── pyproject.toml           package metadata and dependency ranges
└── requirements.txt        beginner-friendly installation entry point
```

## Basic workflow

1. Generate a compatible Complete Audit in VerseVAD.
2. Copy its ZIP into `source/`.
3. Open PowerShell or Terminal in `PostAnalysisToolkit`.
4. Activate `.venv`.
5. Run the desired script, for example:

   ```text
   python scripts/correlation.py
   ```

6. Follow the validated interactive prompts.
7. Open the new timestamped run under `exports/`.

Every analytical script also supports `--help` and `--version`.

## Multiple-selection syntax

Prompts that permit several choices use one shared grammar:

| Entry | Meaning |
|---|---|
| `A` | all choices, when allowed |
| `1` | choice 1 |
| `1,2` | choices 1 and 2 |
| `1, 3, 5` | choices 1, 3, and 5 |
| `5-6` | choices 5 through 6 |
| `1,3,5-6` | choices 1, 3, 5, and 6 |

## Outputs and reproducibility

Depending on the tool, a run can contain Excel workbooks, CSV evidence tables,
an `analysis_spec.json` recording user decisions, and an
`analysis_metadata.json` recording source fingerprints, software versions,
sample sizes, and fixed random seeds. Missing values remain missing; they are
not silently replaced with zero. Exact poem/work-level evidence is retained
where the workflow requires it.

See [Methodology](docs/METHODOLOGY.md) for the shared statistical rules and
[Examples](docs/EXAMPLES.md) for typical workflows.

## Methodological caution

The analyst remains responsible for selecting meaningful measurements,
checking coverage and assumptions, defining a defensible comparison family,
and interpreting results in literary and historical context. A correlation is
not causation; a confidence interval is not a literary verdict; an anomaly is
not necessarily an error; and a large effect size does not by itself establish
scholarly importance.

## License

VerseVAD Post-Analysis Toolkit is free and open-source software distributed
under the [GNU General Public License version 3](LICENSE). You may use,
study, modify, and redistribute it under the terms of that license.

## Repository

[https://github.com/VerseVAD/PostAnalysisToolkit](https://github.com/VerseVAD/PostAnalysisToolkit)
