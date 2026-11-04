"""Gates that must pass before a commit exists: tests, lint, whitespace."""

import sys
from pathlib import Path

import pytest

from tests.conftest import run_git
from tools.gates import GateError, run_lint, run_tests, whitespace_check
from tools.policy import Policy

PASS = (sys.executable, "-c", "import sys; sys.exit(0)")
FAIL = (sys.executable, "-c", "import sys; print('boom'); sys.exit(3)")


def test_passing_test_command_is_silent(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_tests(repo, Policy(test_command=PASS))

    assert "$ " in capsys.readouterr().out


def test_failing_test_command_refuses_the_commit(repo: Path) -> None:
    with pytest.raises(GateError, match="tests failed"):
        run_tests(repo, Policy(test_command=FAIL))


def test_missing_test_executable_is_reported_not_crashed(repo: Path) -> None:
    with pytest.raises(GateError, match="could not run"):
        run_tests(repo, Policy(test_command=("/no/such/python", "-m", "pytest")))


def test_relative_test_command_runs_from_the_repository_root(repo: Path) -> None:
    script = repo / "bin" / "check"
    script.parent.mkdir()
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(0o755)

    run_tests(repo, Policy(test_command=("bin/check",)))


def test_lint_is_skipped_when_no_command_is_configured(repo: Path) -> None:
    run_lint(repo, Policy(lint_command=()))


def test_failing_lint_refuses_the_commit(repo: Path) -> None:
    with pytest.raises(GateError, match="lint failed"):
        run_lint(repo, Policy(lint_command=FAIL))


def test_clean_tree_passes_whitespace_check(repo: Path) -> None:
    whitespace_check(repo)


def test_trailing_whitespace_in_a_tracked_change_is_refused(repo: Path) -> None:
    (repo / "README.rst").write_text("Fixture repository   \n")

    with pytest.raises(GateError, match="whitespace or conflict markers"):
        whitespace_check(repo)


def test_conflict_markers_in_a_staged_change_are_refused(repo: Path) -> None:
    (repo / "README.rst").write_text("<<<<<<< HEAD\nours\n=======\ntheirs\n>>>>>>> other\n")
    run_git(repo, "add", "README.rst")

    with pytest.raises(GateError, match="whitespace or conflict markers"):
        whitespace_check(repo)


def test_untracked_files_are_outside_the_whitespace_check(repo: Path) -> None:
    (repo / "new.py").write_text("x = 1   \n")

    whitespace_check(repo)
