"""Thin ``git`` wrapper used by every tool in this package.

All functions take the repository root explicitly; nothing here consults the
current working directory. Failures carry git's stderr so the caller can
report them verbatim.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


class GitError(RuntimeError):
    """A git command failed or a precondition on the repository is unmet."""


def git(repo: Path, *args: str, check: bool = True) -> str:
    """Run ``git`` in ``repo`` and return its stripped stdout.

    Raises:
        GitError: ``check`` is true and the command exited non-zero.
    """
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout.strip()


def local_identity(repo: Path) -> tuple[str, str]:
    """Return the repository-local ``user.name`` and ``user.email``.

    The global and system configuration are never consulted: an agent must
    commit as the identity the maintainer configured for this checkout, and
    nothing else.

    Raises:
        GitError: either value is unset in the local configuration.
    """
    name = git(repo, "config", "--local", "user.name", check=False)
    email = git(repo, "config", "--local", "user.email", check=False)
    if not name or not email:
        raise GitError(
            "local Git identity is not set. Ask the maintainer for their preferred\n"
            "name and verified email, then run:\n"
            '  git config --local user.name "<name>"\n'
            "  git config --local user.email <email>\n"
            "Do not fall back to the global config and do not invent an identity."
        )
    return name, email


def current_branch(repo: Path) -> str:
    """Return the checked-out branch name (``HEAD`` when detached)."""
    return git(repo, "rev-parse", "--abbrev-ref", "HEAD")


def tracked_paths(repo: Path) -> list[str]:
    """Return every path in the index, sorted as ``git ls-files`` reports it."""
    return [line for line in git(repo, "ls-files").splitlines() if line]


def head_paths(repo: Path) -> list[str]:
    """Return every path in the ``HEAD`` commit; empty when there is no commit yet."""
    listing = git(repo, "ls-tree", "-r", "--name-only", "HEAD", check=False)
    return [line for line in listing.splitlines() if line]
