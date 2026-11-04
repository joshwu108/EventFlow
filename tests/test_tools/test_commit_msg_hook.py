"""The commit-msg hook enforces the single-author rule on every commit in the checkout."""

import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import run_git
from tools.commit_msg_hook import HOOK_MARKER, check_message_file, install_hook, main


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "COMMIT_EDITMSG"
    path.write_text(text)
    return path


def test_plain_message_passes(tmp_path: Path) -> None:
    check_message_file(write(tmp_path, "feat: thing\n\nBody.\n"))


def test_co_author_trailer_fails_with_exit_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write(tmp_path, "feat: thing\n\nCo-Authored-By: Bot <bot@example.invalid>\n")

    assert main([str(path)]) == 1
    assert "single author" in capsys.readouterr().err


def test_git_comment_lines_are_ignored(tmp_path: Path) -> None:
    path = write(tmp_path, "feat: thing\n# Co-Authored-By: shown in the template only\n")

    assert main([str(path)]) == 0


def test_install_writes_an_executable_marked_hook(repo: Path) -> None:
    hook = install_hook(repo, sys.executable)

    assert hook == repo / ".git" / "hooks" / "commit-msg"
    assert HOOK_MARKER in hook.read_text()
    assert hook.stat().st_mode & 0o111


def test_install_refuses_to_overwrite_a_foreign_hook(repo: Path) -> None:
    foreign = repo / ".git" / "hooks" / "commit-msg"
    foreign.parent.mkdir(parents=True, exist_ok=True)
    foreign.write_text("#!/bin/sh\nexit 0\n")

    with pytest.raises(FileExistsError, match="not installed by EventFlow"):
        install_hook(repo, sys.executable)
    assert foreign.read_text() == "#!/bin/sh\nexit 0\n"


def test_install_is_idempotent(repo: Path) -> None:
    first = install_hook(repo, sys.executable).read_text()

    assert install_hook(repo, sys.executable).read_text() == first


def test_installed_hook_blocks_a_plain_git_commit_with_a_trailer(repo: Path) -> None:
    tools_src = Path(__file__).resolve().parents[2] / "tools"
    (repo / "tools").symlink_to(tools_src)
    install_hook(repo, sys.executable)
    (repo / "a.py").write_text("a\n")
    run_git(repo, "add", "a.py")

    blocked = subprocess.run(
        ["git", "commit", "-q", "-m", "feat: a\n\nCo-Authored-By: Bot <bot@example.invalid>"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    allowed = subprocess.run(
        ["git", "commit", "-q", "-m", "feat: a"], cwd=repo, capture_output=True, text=True
    )

    assert blocked.returncode != 0
    assert "single author" in blocked.stderr
    assert allowed.returncode == 0
