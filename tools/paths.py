"""Paths the commit tooling must never stage.

Three groups of rules, checked in order:

* ``PRIVATE_PATTERNS`` and ``PRIVATE_DIRECTORIES`` cover Markdown, planning
  documents, local agent configuration, environments and secrets. They have
  no exceptions.
* ``DATASET_PATTERNS`` cover downloaded or generated data. Small files under
  ``FIXTURE_PREFIX`` are exempt, up to ``MAX_FIXTURE_BYTES``, so a fresh
  clone can run tests that read a committed excerpt without ever carrying a
  full archive.

Matching is case-insensitive so ``README.MD`` is treated like ``README.md``.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path, PurePosixPath

PRIVATE_PATTERNS: tuple[str, ...] = (
    "*.md",
    "*.markdown",
    "*.mdx",
    "*_prd.*",
    "*.prd",
    "*.plan",
    ".env",
    ".env.*",
    "*.pem",
)

# A path is forbidden when any of its directory components is one of these.
PRIVATE_DIRECTORIES: frozenset[str] = frozenset(
    {"planning", ".planning", "plans", ".claude", ".agents", ".venv", "venv"}
)

DATASET_PATTERNS: tuple[str, ...] = (
    "*.zip",
    "*.npy",
    "*.parquet",
    "*.feather",
    "*.csv.gz",
    "*.csv",
)
DATASET_DIRECTORIES: frozenset[str] = frozenset({"data"})

FIXTURE_PREFIX = "tests/fixtures/"
MAX_FIXTURE_BYTES = 64 * 1024  # Largest file accepted under the fixture exception.


def _matches(name: str, path: str, patterns: tuple[str, ...]) -> str | None:
    for pattern in patterns:
        if fnmatch.fnmatchcase(name, pattern) or fnmatch.fnmatchcase(path, pattern):
            return pattern
    return None


def forbidden_reason(repo: Path, path: str) -> str | None:
    """Return why ``path`` may not be staged, or ``None`` when it may.

    Args:
        repo: Repository root; only consulted to size files under the
            fixture exception.
        path: Repository-relative path using ``/`` separators.
    """
    lowered = path.lower()
    posix = PurePosixPath(lowered)
    name = posix.name
    directories = posix.parts[:-1]

    for directory in directories:
        if directory in PRIVATE_DIRECTORIES:
            return f"directory '{directory}'"
    private = _matches(name, lowered, PRIVATE_PATTERNS)
    if private is not None:
        return f"pattern '{private}'"

    if lowered.startswith(FIXTURE_PREFIX):
        target = repo / path
        size = target.stat().st_size if target.exists() else 0
        if size > MAX_FIXTURE_BYTES:
            return f"fixture over {MAX_FIXTURE_BYTES} bytes ({size})"
        return None

    for directory in directories:
        if directory in DATASET_DIRECTORIES:
            return f"dataset directory '{directory}'"
    dataset = _matches(name, lowered, DATASET_PATTERNS)
    if dataset is not None:
        return f"dataset pattern '{dataset}'"
    return None
