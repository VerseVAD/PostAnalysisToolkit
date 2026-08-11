"""Shared filesystem, naming, hashing, and serialization helpers."""

from __future__ import annotations

import hashlib
import math
import re
import sys
from pathlib import Path
from typing import Any


def configure_console_encoding() -> None:
    """Prevent non-ASCII status text from crashing legacy Windows consoles."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError, ValueError):
                pass


def project_root(script_file: str | Path) -> Path:
    return Path(script_file).resolve().parent.parent


def source_directory(root: Path) -> Path:
    return root / "source"


def export_directory(root: Path, tool_name: str) -> Path:
    return root / "exports" / tool_name


def slugify(text: str, *, fallback: str = "analysis", max_length: int = 72) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip()).strip("_").lower()
    return (value or fallback)[:max_length]


def pretty_words(value: str) -> str:
    text = re.sub(r"[._]+", " ", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    return " ".join(
        word.upper() if word.lower() in {"vad", "aoa", "sd", "iqr"} else word.capitalize()
        for word in text.split()
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    """Convert common scientific-Python values to strict JSON values."""

    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            value = item_method()
        except (TypeError, ValueError):
            pass
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def public_source_reference(path: Path, root: Path) -> str:
    """Return a reproducible label without leaking a user's home path."""

    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return Path(path).name
