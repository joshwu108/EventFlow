"""The commit policy is read from TOML and validated before anything trusts it."""

from pathlib import Path

import pytest

from tools import policy as policy_mod
from tools.policy import Policy, PolicyError, load_policy

GRANT = """
auto_commit = true
auto_push = true
granted_on = "2026-11-04"
granted_by = "maintainer"
remote = "origin"
branch = "main"
test_command = ["python", "-m", "pytest", "-q"]
lint_command = ["ruff", "check", "."]
requires_explicit_approval = ["merge", "rebase"]
protected_paths = ["src/secret.py"]
"""


def write_policy(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "commit_policy.toml"
    path.write_text(text)
    return path


def test_load_policy_reads_every_grant_field(tmp_path: Path) -> None:
    loaded = load_policy(write_policy(tmp_path, GRANT))

    assert loaded == Policy(
        auto_commit=True,
        auto_push=True,
        granted_on="2026-11-04",
        granted_by="maintainer",
        remote="origin",
        branch="main",
        test_command=("python", "-m", "pytest", "-q"),
        lint_command=("ruff", "check", "."),
        requires_explicit_approval=("merge", "rebase"),
        protected_paths=("src/secret.py",),
    )


def test_missing_policy_file_grants_nothing(tmp_path: Path) -> None:
    loaded = load_policy(tmp_path / "absent.toml")

    assert loaded.auto_commit is False
    assert loaded.auto_push is False
    assert loaded.granted_by == ""


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(PolicyError, match="unknown key"):
        load_policy(write_policy(tmp_path, GRANT + 'surprise = "yes"\n'))


def test_non_boolean_flag_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(PolicyError, match="auto_commit"):
        load_policy(
            write_policy(tmp_path, GRANT.replace("auto_commit = true", 'auto_commit = "yes"'))
        )


def test_push_grant_without_commit_grant_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(PolicyError, match="auto_push"):
        load_policy(
            write_policy(tmp_path, GRANT.replace("auto_commit = true", "auto_commit = false"))
        )


def test_empty_test_command_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(PolicyError, match="test_command"):
        load_policy(write_policy(tmp_path, GRANT.replace('["python", "-m", "pytest", "-q"]', "[]")))


def test_policy_is_immutable(tmp_path: Path) -> None:
    loaded = load_policy(write_policy(tmp_path, GRANT))
    with pytest.raises(AttributeError):
        loaded.auto_push = False  # type: ignore[misc]


def test_repository_policy_grants_commit_and_push_to_origin_main() -> None:
    loaded = load_policy(policy_mod.DEFAULT_POLICY_PATH)

    assert loaded.auto_commit and loaded.auto_push
    assert (loaded.remote, loaded.branch) == ("origin", "main")
    assert "force-push" in loaded.requires_explicit_approval
    assert loaded.granted_by == "joshuawu"
