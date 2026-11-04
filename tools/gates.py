"""Checks that must pass before a commit is created.

Each gate either returns quietly or raises ``GateError`` with the reason.
Commands run from the repository root with inherited stdio so their own
output stays visible.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from tools.policy import Policy


class GateError(RuntimeError):
    """A gate failed; the commit must not proceed."""


def _run(repo: Path, command: tuple[str, ...], label: str) -> None:
    print(f"$ {' '.join(command)}", flush=True)
    try:
        result = subprocess.run(command, cwd=repo, check=False)
    except OSError as exc:
        raise GateError(f"could not run the {label} command {command[0]!r}: {exc}") from exc
    if result.returncode != 0:
        raise GateError(
            f"{label} failed (exit {result.returncode}): refusing to commit a broken tree."
        )


def run_tests(repo: Path, policy: Policy) -> None:
    """Run ``policy.test_command``; raise ``GateError`` unless it exits 0."""
    _run(repo, policy.test_command, "tests")


def run_lint(repo: Path, policy: Policy) -> None:
    """Run ``policy.lint_command`` when one is configured."""
    if policy.lint_command:
        _run(repo, policy.lint_command, "lint")


def whitespace_check(repo: Path) -> None:
    """Refuse trailing whitespace and conflict markers in tracked or staged changes.

    Untracked files are not inspected here; they are checked once staged,
    because ``git diff --check`` only sees the index and the working tree
    of tracked paths.
    """
    for scope in ((), ("--cached",)):
        result = subprocess.run(
            ["git", "diff", *scope, "--check"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise GateError(
                "git diff --check reported whitespace or conflict markers:\n"
                + (result.stdout or result.stderr).strip()
            )
