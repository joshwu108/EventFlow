"""End-to-end: commit_safe creates exactly the requested commit or nothing."""

import sys
from pathlib import Path

import pytest

from tests.conftest import IDENTITY_EMAIL, IDENTITY_NAME, run_git
from tools import commit_safe

PASS = f'["{sys.executable}", "-c", "import sys; sys.exit(0)"]'
FAIL = f'["{sys.executable}", "-c", "import sys; sys.exit(3)"]'


def write_policy(repo: Path, *, auto_commit: bool = True, test_command: str = PASS) -> Path:
    path = repo / "policy.toml"
    path.write_text(
        f"auto_commit = {str(auto_commit).lower()}\n"
        'granted_on = "2026-11-04"\ngranted_by = "maintainer"\n'
        f"test_command = {test_command}\n"
    )
    return path


def run(repo: Path, policy: Path, *args: str) -> int:
    return commit_safe.main(["--repo", str(repo), "--policy", str(policy), *args])


def head_count(repo: Path) -> int:
    return int(run_git(repo, "rev-list", "--count", "HEAD"))


def test_commit_is_created_with_the_local_identity_and_message(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    policy = write_policy(repo)
    (repo / "a.py").write_text("a = 1\n")

    status = run(repo, policy, "-m", "feat: add a", "--body", "Because.", "--path", "a.py")

    assert status == 0
    assert head_count(repo) == 2
    assert run_git(repo, "log", "-1", "--format=%an <%ae>") == f"{IDENTITY_NAME} <{IDENTITY_EMAIL}>"
    assert run_git(repo, "log", "-1", "--format=%B").strip() == "feat: add a\n\nBecause."
    assert run_git(repo, "diff", "--cached", "--name-only") == ""
    assert "committed locally" in capsys.readouterr().out


def test_dry_run_commits_nothing_and_leaves_the_index_empty(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    policy = write_policy(repo)
    (repo / "a.py").write_text("a = 1\n")

    status = run(repo, policy, "-m", "feat: add a", "--path", "a.py", "--dry-run")

    assert status == 0
    assert head_count(repo) == 1
    assert run_git(repo, "diff", "--cached", "--name-only") == ""
    assert "dry run" in capsys.readouterr().out


def test_unchanged_allowlist_is_a_no_op(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    policy = write_policy(repo)

    status = run(repo, policy, "-m", "docs: nothing", "--path", "README.rst")

    assert status == 0
    assert head_count(repo) == 1
    assert "nothing to commit" in capsys.readouterr().out


def test_failing_tests_commit_nothing(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    policy = write_policy(repo, test_command=FAIL)
    (repo / "a.py").write_text("a = 1\n")

    status = run(repo, policy, "-m", "feat: add a", "--path", "a.py")

    assert status == 1
    assert head_count(repo) == 1
    assert run_git(repo, "diff", "--cached", "--name-only") == ""
    assert "ABORTED: tests failed" in capsys.readouterr().err


def test_skip_tests_is_explicit_and_announced(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    policy = write_policy(repo, test_command=FAIL)
    (repo / "a.py").write_text("a = 1\n")

    status = run(repo, policy, "-m", "feat: add a", "--path", "a.py", "--skip-tests")

    assert status == 0
    assert head_count(repo) == 2
    assert "tests skipped" in capsys.readouterr().out


def test_forbidden_path_aborts_before_any_gate_runs(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    policy = write_policy(repo, test_command=FAIL)
    (repo / "notes.md").write_text("x\n")

    status = run(repo, policy, "-m", "docs: notes", "--path", "notes.md")

    assert status == 1
    assert "refusing notes.md" in capsys.readouterr().err
    assert head_count(repo) == 1


def test_co_author_trailer_aborts(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    policy = write_policy(repo)
    (repo / "a.py").write_text("a = 1\n")

    status = run(
        repo, policy, "-m", "feat: add a", "--body", "Co-Authored-By: X <x@y>", "--path", "a.py"
    )

    assert status == 1
    assert "single author" in capsys.readouterr().err
    assert head_count(repo) == 1


def test_without_a_grant_every_commit_is_refused(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    policy = write_policy(repo, auto_commit=False)
    (repo / "a.py").write_text("a = 1\n")

    status = run(repo, policy, "-m", "feat: add a", "--path", "a.py")

    assert status == 1
    assert "ask before every commit" in capsys.readouterr().err


def test_missing_local_identity_aborts(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    policy = write_policy(repo)
    run_git(repo, "config", "--local", "--unset", "user.name")
    (repo / "a.py").write_text("a = 1\n")

    status = run(repo, policy, "-m", "feat: add a", "--path", "a.py")

    assert status == 1
    assert "local Git identity is not set" in capsys.readouterr().err


def test_whitespace_errors_abort(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    policy = write_policy(repo)
    (repo / "README.rst").write_text("Fixture repository   \n")

    status = run(repo, policy, "-m", "docs: trailing", "--path", "README.rst")

    assert status == 1
    assert "whitespace" in capsys.readouterr().err
