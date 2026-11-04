"""The ``repo`` fixture provides exactly the preconditions the tools rely on."""

from pathlib import Path

from tests.conftest import IDENTITY_EMAIL, IDENTITY_NAME, run_git


def test_repo_fixture_is_on_main_with_one_commit(repo: Path) -> None:
    assert run_git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert run_git(repo, "rev-list", "--count", "HEAD") == "1"


def test_repo_fixture_sets_a_local_identity_only(repo: Path) -> None:
    assert run_git(repo, "config", "--local", "user.name") == IDENTITY_NAME
    assert run_git(repo, "config", "--local", "user.email") == IDENTITY_EMAIL


def test_repo_fixture_ignores_markdown_and_planning(repo: Path) -> None:
    (repo / "planning").mkdir()
    (repo / "planning" / "notes.md").write_text("private\n")
    (repo / "NOTES.md").write_text("private\n")
    assert run_git(repo, "status", "--porcelain") == ""
