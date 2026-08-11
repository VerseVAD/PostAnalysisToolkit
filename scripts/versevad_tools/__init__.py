"""Shared infrastructure for the VerseVAD Post-Analysis Toolkit.

The analytical scripts keep their own version numbers because those versions
identify the exact statistical engine used for a run. This package contains
only shared command-line, path, source-discovery, labeling, and serialization
helpers; it does not perform statistical analysis.
"""

from __future__ import annotations

__version__ = "0.1.0"
