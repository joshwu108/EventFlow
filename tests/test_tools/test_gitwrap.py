"""Thin git wrapper: explicit repository, local-only identity, loud failures."""

from pathlib import Path

import pytest

from tests.conftest import IDENTITY_EMAIL, IDENTITY_NAME, run_git
from tools.gitwrap import GitError, current_branch, git, local_identity, tracked_paths


def test_git_returns_stripped_stdout(repo: Path) -> None:
    assert git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"


def test_git_raises_with_stderr_on_failure(repo: Path) -> None:
    with pytest.raises(GitError, match="unknown revision|bad revision"):
        git(repo, "rev-parse", "--verify", "no-such-ref^{commit}")


def test_git_can_be_asked_not_to_raise(repo: Path) -> None:
    assert git(repo, "config", "--local", "no.such.key", check=False) == ""


def test_local_identity_reads_only_the_repository_config(repo: Path) -> None:
    assert local_identity(repo) == (IDENTITY_NAME, IDENTITY_EMAIL)


def test_missing_local_identity_is_an_error_even_with_global_config(
    repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A global identity must never be used as a fallback.
    global_config = tmp_path / "gitconfig"
    global_config.write_text("[user]\n\tname = Global Person\n\temail = global@example.invalid\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))
    run_git(repo, "config", "--local", "--unset", "user.email")

    with pytest.raises(GitError, match="local Git identity is not set"):
        local_identity(repo)


def test_current_branch_names_the_checked_out_branch(repo: Path) -> None:
    assert current_branch(repo) == "main"
    run_git(repo, "switch", "-q", "-c", "topic")
    assert current_branch(repo) == "topic"


def test_tracked_paths_lists_the_index(repo: Path) -> None:
    assert tracked_paths(repo) == [".gitignore", "README.rst"]
