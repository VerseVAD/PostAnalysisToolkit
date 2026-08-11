# Getting Started

This guide assumes you know what Python is but may not have used it from a
terminal. The toolkit has no graphical interface: you run a script, answer
plain-language prompts, and open the generated Excel/CSV/JSON files afterward.

## 1. Open a terminal

- **Windows:** open Start, search for **PowerShell**, and open it.
- **macOS:** open Applications → Utilities → **Terminal**.
- **Linux:** open your distribution's Terminal application.

## 2. Check Python

Windows:

```powershell
python --version
```

macOS/Linux:

```bash
python3 --version
```

Use Python 3.10–3.14. If the command is missing or reports an older version,
install Python from [python.org](https://www.python.org/downloads/). On Windows,
allow the installer to add Python to PATH. Then close and reopen the terminal.

## 3. Download the repository

If Git is installed:

```text
git clone https://github.com/VerseVAD/PostAnalysisToolkit.git
cd PostAnalysisToolkit
```

Alternatively, use GitHub's **Code → Download ZIP**, extract it, and navigate
to the extracted `PostAnalysisToolkit` folder with `cd`.

## 4. Create an isolated environment

A virtual environment keeps toolkit packages separate from other software.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation for this window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

When active, the prompt normally begins with `(.venv)`.

## 5. Install dependencies

On every platform:

```text
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Confirm the installation:

```text
python scripts/correlation.py --version
python scripts/correlation.py --help
```

## 6. Add a VerseVAD audit

Generate a Complete Audit in VerseVAD and copy the ZIP into `source/`. Corpus
tools use corpus audits; `single.py` uses a single-poem audit-schema-v2 ZIP.
Do not unzip the audit unless a tool explicitly requests a standalone CSV.

The repository ignores `source/*`, so personal research files will not be
included in an ordinary Git commit.

## 7. Run an analysis

Windows example:

```powershell
python scripts\correlation.py
```

macOS/Linux example:

```bash
python scripts/correlation.py
```

Answer the prompts. Selection fields accept `A`, `1`, `1,2`, `1, 3, 5`,
`5-6`, and combinations such as `1,3,5-6`.

## 8. Find the results

The terminal prints the exact output folder at completion. It will be a
timestamped folder under `exports/<tool>/`. Excel files are designed for human
review; CSV and JSON files preserve detailed evidence and reproducibility data.

## Returning later

Open a terminal in the repository and reactivate the environment before use:

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

To update a clean clone:

```text
git pull
python -m pip install -r requirements.txt
```

Your ignored `source/` audits and `exports/` runs remain local.

## Common problems

- **No source found:** place the correct Complete Audit ZIP in `source/` or use
  the script's documented `--source` option.
- **Incompatible audit:** regenerate a current Complete Audit in VerseVAD; do
  not rename an unrelated ZIP to make it appear compatible.
- **Missing package:** reactivate `.venv` and rerun the requirements command.
- **Permission error:** choose a writable repository/output folder and ensure
  an exported workbook is not already open when the script replaces it.
- **Bad ZIP:** download/export the audit again; the toolkit will not guess past
  a corrupted archive.
