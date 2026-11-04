"""Pushes are fast-forward only, to the configured branch, after a clean commit."""

from pathlib import Path

import pytest

from tests.conftest import clone_of, run_git
from tools.policy import Policy
from tools.push import PushError, push_milestone

GRANTED = Policy(auto_commit=True, auto_push=True, remote="origin", branch="main")


def commit_file(repo: Path, name: str) -> str:
    (repo / name).write_text(f"{name}\n")
    run_git(repo, "add", name)
    run_git(repo, "commit", "-q", "-m", f"feat: {name}")
    return run_git(repo, "rev-parse", "HEAD")


def remote_head(remote: Path) -> str:
    return run_git(remote, "rev-parse", "refs/heads/main")


def test_new_commits_are_pushed_and_reported(repo: Path, remote: Path) -> None:
    commit_file(repo, "a.py")
    head = commit_file(repo, "b.py")

    report = push_milestone(repo, GRANTED)

    assert report == "pushed 2 commits to origin/main"
    assert remote_head(remote) == head


def test_nothing_to_push_is_a_no_op(repo: Path, remote: Path) -> None:
    assert push_milestone(repo, GRANTED) == "nothing to push: origin/main is at HEAD"


def test_push_without_a_push_grant_is_refused(repo: Path, remote: Path) -> None:
    commit_file(repo, "a.py")

    with pytest.raises(PushError, match="no push grant"):
        push_milestone(repo, Policy(auto_commit=True, auto_push=False))


def test_push_from_another_branch_is_refused(repo: Path, remote: Path) -> None:
    run_git(repo, "switch", "-q", "-c", "topic")
    commit_file(repo, "a.py")

    with pytest.raises(PushError, match="on branch 'topic', not 'main'"):
        push_milestone(repo, GRANTED)
    assert remote_head(remote) == run_git(repo, "rev-parse", "main")


def test_diverged_remote_is_refused_never_forced(repo: Path, remote: Path, tmp_path: Path) -> None:
    other = clone_of(remote, tmp_path, "other")
    theirs = commit_file(other, "theirs.py")
    run_git(other, "push", "-q", "origin", "main")
    commit_file(repo, "ours.py")

    with pytest.raises(PushError, match="behind origin/main"):
        push_milestone(repo, GRANTED)
    assert remote_head(remote) == theirs


def test_staged_changes_block_the_push(repo: Path, remote: Path) -> None:
    commit_file(repo, "a.py")
    (repo / "b.py").write_text("b\n")
    run_git(repo, "add", "b.py")

    with pytest.raises(PushError, match="index is not empty"):
        push_milestone(repo, GRANTED)


def test_missing_remote_is_refused(repo: Path) -> None:
    commit_file(repo, "a.py")

    with pytest.raises(PushError, match="remote 'origin'"):
        push_milestone(repo, GRANTED)
