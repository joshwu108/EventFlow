"""Validate the explicit allowlist of paths a commit may contain.

The tooling never stages ``.``, ``-A`` or a glob: every path is named, must
be a single file inside the repository, must not match a forbidden rule
(``tools.paths``), and must exist in the working tree or already be tracked
(so deletions can be committed). Paths the policy protects additionally need
the maintainer's explicit approval.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from tools.gitwrap import head_paths, tracked_paths
from tools.paths import forbidden_reason
from tools.policy import Policy

WILDCARD_CHARACTERS = ("*", "?", "[")


class AllowlistError(ValueError):
    """The allowlist cannot be staged as given; nothing has been touched."""


def _normalise(raw: str) -> str:
    path = raw.replace("\\", "/")
    if any(char in path for char in WILDCARD_CHARACTERS) or path in (".", "") or path.endswith("/"):
        raise AllowlistError(f"refusing {raw!r}: not a single file path")
    posix = PurePosixPath(path)
    if posix.is_absolute() or ".." in posix.parts:
        raise AllowlistError(f"refusing {raw!r}: must be a relative path inside the repository")
    return str(posix)


def check_allowlist(
    repo: Path, paths: list[str], policy: Policy, *, approve_protected: bool = False
) -> list[str]:
    """Return the normalised allowlist after every check passes.

    Raises:
        AllowlistError: the list is empty, contains a duplicate, a wildcard,
            an absolute or escaping path, a forbidden path, a path that is
            neither present nor tracked, or a protected path without
            ``approve_protected``.
    """
    if not paths:
        raise AllowlistError("no --path given: this tool never stages wildcards.")
    normalised = [_normalise(path) for path in paths]
    seen: set[str] = set()
    tracked: set[str] | None = None
    for path in normalised:
        if path in seen:
            raise AllowlistError(f"refusing {path}: listed twice")
        seen.add(path)
        reason = forbidden_reason(repo, path)
        if reason is not None:
            raise AllowlistError(f"refusing {path}: {reason}")
        if (repo / path).is_dir():
            raise AllowlistError(f"refusing {path!r}: not a single file path")
        if not (repo / path).exists():
            if tracked is None:
                tracked = set(tracked_paths(repo)) | set(head_paths(repo))
            if path not in tracked:
                raise AllowlistError(f"refusing {path}: not present in the working tree")
        if path in policy.protected_paths and not approve_protected:
            raise AllowlistError(
                f"{path} is protected by the commit policy. Re-run with --approve-protected\n"
                "only after the maintainer has explicitly approved the change."
            )
    return normalised
