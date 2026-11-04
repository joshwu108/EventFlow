"""An explicit allowlist is the only way paths reach the index."""

from pathlib import Path

import pytest

from tests.conftest import run_git
from tools.allowlist import AllowlistError, check_allowlist
from tools.policy import Policy


def touch(repo: Path, rel: str, text: str = "x\n") -> None:
    target = repo / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


def test_existing_source_and_test_paths_pass(repo: Path) -> None:
    touch(repo, "src/pkg/mod.py")
    touch(repo, "tests/test_mod.py")

    check_allowlist(repo, ["src/pkg/mod.py", "tests/test_mod.py"], Policy())


def test_empty_allowlist_is_refused(repo: Path) -> None:
    with pytest.raises(AllowlistError, match="never stages wildcards"):
        check_allowlist(repo, [], Policy())


@pytest.mark.parametrize("path", ["src/*.py", "tests/?", ".", "src/"])
def test_wildcards_dots_and_directories_are_refused(repo: Path, path: str) -> None:
    (repo / "src").mkdir()
    with pytest.raises(AllowlistError, match="not a single file path"):
        check_allowlist(repo, [path], Policy())


@pytest.mark.parametrize("path", ["/etc/passwd", "../outside.py", "src/../../x.py"])
def test_absolute_and_escaping_paths_are_refused(repo: Path, path: str) -> None:
    with pytest.raises(AllowlistError, match="inside the repository"):
        check_allowlist(repo, [path], Policy())


def test_forbidden_path_is_refused_with_its_rule(repo: Path) -> None:
    touch(repo, "notes.md")
    with pytest.raises(AllowlistError, match=r"refusing notes.md: pattern '\*\.md'"):
        check_allowlist(repo, ["notes.md"], Policy())


def test_missing_untracked_path_is_refused(repo: Path) -> None:
    with pytest.raises(AllowlistError, match="not present in the working tree"):
        check_allowlist(repo, ["ghost.py"], Policy())


def test_deleted_tracked_path_passes(repo: Path) -> None:
    run_git(repo, "rm", "-q", "README.rst")

    check_allowlist(repo, ["README.rst"], Policy())


def test_duplicate_entries_are_refused(repo: Path) -> None:
    touch(repo, "a.py")
    with pytest.raises(AllowlistError, match="listed twice"):
        check_allowlist(repo, ["a.py", "a.py"], Policy())


def test_protected_path_needs_explicit_approval(repo: Path) -> None:
    touch(repo, "src/kernel.py")
    policy = Policy(protected_paths=("src/kernel.py",))

    with pytest.raises(AllowlistError, match="--approve-protected"):
        check_allowlist(repo, ["src/kernel.py"], policy)
    check_allowlist(repo, ["src/kernel.py"], policy, approve_protected=True)


def test_backslash_separators_are_normalised(repo: Path) -> None:
    touch(repo, "src/pkg/mod.py")

    assert check_allowlist(repo, ["src\\pkg\\mod.py"], Policy()) == ["src/pkg/mod.py"]
