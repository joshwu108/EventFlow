"""Shared fixtures: throwaway Git repositories with a local identity."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

IDENTITY_NAME = "Test Author"
IDENTITY_EMAIL = "test-author@example.invalid"


def run_git(repo: Path, *args: str) -> str:
    """Run ``git`` in ``repo`` and return its stripped stdout, raising on failure."""
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)
    return result.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """An initialised repository on ``main`` with a local identity and one commit.

    The global and system Git configuration are not consulted by the tools,
    but the fixture sets the identity locally so that commits made through
    the fixture itself succeed everywhere.
    """
    root = tmp_path / "repo"
    root.mkdir()
    run_git(root, "init", "-q", "-b", "main")
    run_git(root, "config", "--local", "user.name", IDENTITY_NAME)
    run_git(root, "config", "--local", "user.email", IDENTITY_EMAIL)
    run_git(root, "config", "--local", "commit.gpgsign", "false")
    (root / ".gitignore").write_text("*.md\nplanning/\n.claude/\n.venv/\n")
    (root / "README.rst").write_text("Fixture repository\n")
    run_git(root, "add", ".gitignore", "README.rst")
    run_git(root, "commit", "-q", "-m", "chore: initial")
    return root


@pytest.fixture
def remote(repo: Path, tmp_path: Path) -> Path:
    """A bare ``origin`` that already holds the fixture repository's ``main``."""
    bare = tmp_path / "origin.git"
    run_git(tmp_path, "init", "-q", "--bare", "-b", "main", str(bare))
    run_git(repo, "remote", "add", "origin", str(bare))
    run_git(repo, "push", "-q", "-u", "origin", "main")
    return bare


def clone_of(remote: Path, tmp_path: Path, name: str) -> Path:
    """A second working clone of ``remote`` with its own local identity."""
    other = tmp_path / name
    run_git(tmp_path, "clone", "-q", str(remote), str(other))
    run_git(other, "config", "--local", "user.name", "Other Person")
    run_git(other, "config", "--local", "user.email", "other@example.invalid")
    run_git(other, "config", "--local", "commit.gpgsign", "false")
    return other
