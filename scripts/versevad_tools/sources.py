"""Discovery and validation helpers for VerseVAD audit sources."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Sequence

from versevad_tools.audit import AuditSourceError, require_audit


def discover_files(directory: Path, suffixes: Sequence[str] = (".zip",)) -> list[Path]:
    allowed = {suffix.casefold() for suffix in suffixes}
    if not directory.exists():
        return []
    return sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.casefold() in allowed],
        key=lambda path: path.name.casefold(),
    )


def discover_corpus_metric_sources(directory: Path, metrics_filename: str) -> list[Path]:
    candidates: list[Path] = []
    for path in discover_files(directory):
        try:
            require_audit(
                path,
                expected_analysis_mode="corpus",
                require_complete=True,
            )
            candidates.append(path)
        except (OSError, zipfile.BadZipFile, AuditSourceError):
            continue
    return candidates


def choose_one_source(paths: Sequence[Path]) -> Path:
    if not paths:
        raise FileNotFoundError("No compatible VerseVAD sources were found in the source folder.")
    if len(paths) == 1:
        print(f"Found one corpus source:\n  {paths[0].name}\n")
        return paths[0]
    print("Available corpus sources")
    print("========================")
    for index, path in enumerate(paths, start=1):
        print(f"[{index}] {path.name}")
    while True:
        raw = input("\nSelect corpus: ").strip()
        try:
            number = int(raw)
        except ValueError:
            print("Enter the number of the corpus you want to analyze.")
            continue
        if 1 <= number <= len(paths):
            print()
            return paths[number - 1]
        print(f"Choose a number from 1 to {len(paths)}.")
