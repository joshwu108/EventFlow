"""The index is rebuilt from the allowlist alone and verified before committing."""

from pathlib import Path

import pytest

from tests.conftest import run_git
from tools.staging import StagingError, reset_index, stage_allowlist, verify_staged


def staged(repo: Path) -> list[str]:
    return run_git(repo, "diff", "--cached", "--name-only").splitlines()


def test_stage_allowlist_stages_exactly_the_listed_paths(repo: Path) -> None:
    (repo / "a.py").write_text("a\n")
    (repo / "b.py").write_text("b\n")

    stage_allowlist(repo, ["a.py"])

    assert staged(repo) == ["a.py"]


def test_stage_allowlist_drops_anything_previously_staged(repo: Path) -> None:
    (repo / "sneaky.py").write_text("s\n")
    run_git(repo, "add", "sneaky.py")
    (repo / "a.py").write_text("a\n")

    stage_allowlist(repo, ["a.py"])

    assert staged(repo) == ["a.py"]


def test_stage_allowlist_stages_a_deletion(repo: Path) -> None:
    (repo / "README.rst").unlink()

    stage_allowlist(repo, ["README.rst"])

    assert run_git(repo, "diff", "--cached", "--name-status") == "D\tREADME.rst"


def test_stage_allowlist_refuses_an_ignored_path_instead_of_forcing_it(repo: Path) -> None:
    (repo / "notes.md").write_text("private\n")

    with pytest.raises(StagingError, match="ignored"):
        stage_allowlist(repo, ["notes.md"])
    assert staged(repo) == []


def test_verify_staged_returns_the_staged_subset(repo: Path) -> None:
    (repo / "a.py").write_text("a\n")
    stage_allowlist(repo, ["a.py", "README.rst"])  # README.rst unchanged: nothing to stage.

    assert verify_staged(repo, ["a.py", "README.rst"]) == ["a.py"]


def test_verify_staged_refuses_paths_outside_the_allowlist(repo: Path) -> None:
    (repo / "a.py").write_text("a\n")
    run_git(repo, "add", "a.py")

    with pytest.raises(StagingError, match="unexpected staged paths"):
        verify_staged(repo, ["README.rst"])


def test_verify_staged_refuses_forbidden_paths_that_reached_the_index(repo: Path) -> None:
    (repo / "notes.md").write_text("private\n")
    run_git(repo, "add", "-f", "notes.md")

    with pytest.raises(StagingError, match="forbidden paths reached the index"):
        verify_staged(repo, ["notes.md"])


def test_reset_index_leaves_nothing_staged_but_keeps_the_working_tree(repo: Path) -> None:
    (repo / "a.py").write_text("a\n")
    run_git(repo, "add", "a.py")

    reset_index(repo)

    assert staged(repo) == []
    assert (repo / "a.py").read_text() == "a\n"
