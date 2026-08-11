"""Consistent command-line parsing used by toolkit scripts."""

from __future__ import annotations

import re
from typing import Optional


def parse_index_selection(
    raw: str,
    n_items: int,
    *,
    min_count: int = 1,
    max_count: Optional[int] = None,
    allow_all: bool = True,
    one_based: bool = False,
) -> list[int]:
    """Parse ``A``, individual choices, ranges, and mixed selections."""

    if n_items < 0:
        raise ValueError("The number of available items cannot be negative.")
    text = str(raw).strip().casefold()
    if text in {"a", "all", "*"}:
        if not allow_all:
            raise ValueError("A/all is not available for this selection.")
        selected = list(range(n_items))
    else:
        chosen: set[int] = set()
        for token in re.split(r"[\s,]+", text):
            if not token:
                continue
            if "-" in token:
                left, right = token.split("-", 1)
                try:
                    start, end = int(left), int(right)
                except ValueError as exc:
                    raise ValueError(f"Invalid selection: {token}") from exc
                lo, hi = sorted((start, end))
                if lo < 1 or hi > n_items:
                    raise ValueError(f"Selection {token} is outside 1-{n_items}.")
                chosen.update(range(lo - 1, hi))
            else:
                try:
                    item = int(token)
                except ValueError as exc:
                    raise ValueError(f"Invalid selection: {token}") from exc
                if not 1 <= item <= n_items:
                    raise ValueError(f"Selection {item} is outside 1-{n_items}.")
                chosen.add(item - 1)
        selected = sorted(chosen)

    if len(selected) < min_count:
        raise ValueError(f"Select at least {min_count} item(s).")
    if max_count is not None and len(selected) > max_count:
        raise ValueError(f"Select no more than {max_count} item(s).")
    return [index + 1 for index in selected] if one_based else selected


def parse_coverage_threshold(
    raw: str,
    *,
    blank: Optional[float] = None,
) -> Optional[float]:
    """Parse either a proportion or percentage as a 0-1 proportion."""

    text = str(raw).strip().rstrip("%").strip()
    if not text:
        return blank
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError("Coverage threshold must be a number such as 80 or 0.80.") from exc
    if value > 1:
        value /= 100.0
    if not 0 <= value <= 1:
        raise ValueError("Coverage threshold must be between 0 and 100%.")
    return value
