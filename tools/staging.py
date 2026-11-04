"""Build the index from the allowlist alone and prove it before committing.

``stage_allowlist`` starts from an empty index so that whatever was staged
earlier, by anyone, cannot ride along. ``verify_staged`` then re-reads the
index and refuses anything outside the allowlist or matching a forbidden
rule. Every failure leaves the index empty and the working tree untouched.
"""

from __future__ import annotations

from pathlib import Path

from tools.gitwrap import GitError, git
from tools.paths import forbidden_reason


class StagingError(RuntimeError):
    """The index does not match the allowlist; it has been reset."""


def reset_index(repo: Path) -> None:
    """Unstage everything without touching the working tree."""
    git(repo, "reset", "-q")


def stage_allowlist(repo: Path, paths: list[str]) -> None:
    """Reset the index and stage ``paths`` (modifications and deletions alike).

    Raises:
        StagingError: git refused a path, typically because it is ignored.
            The tool never force-adds, so an ignored path cannot be staged.
    """
    reset_index(repo)
    try:
        git(repo, "add", "--", *paths)
    except GitError as exc:
        reset_index(repo)
        raise StagingError(f"could not stage the allowlist (ignored path?):\n{exc}") from exc


def verify_staged(repo: Path, paths: list[str]) -> list[str]:
    """Return the staged paths after confirming they are a subset of ``paths``.

    Raises:
        StagingError: the index holds a path outside the allowlist, or a
            forbidden path reached it (for example by an earlier ``add -f``).
    """
    staged = [line for line in git(repo, "diff", "--cached", "--name-only").splitlines() if line]
    unexpected = sorted(set(staged) - set(paths))
    if unexpected:
        reset_index(repo)
        raise StagingError(
            "unexpected staged paths (nothing was committed):\n  " + "\n  ".join(unexpected)
        )
    forbidden = [path for path in staged if forbidden_reason(repo, path)]
    if forbidden:
        reset_index(repo)
        raise StagingError("forbidden paths reached the index: " + ", ".join(forbidden))
    return staged
