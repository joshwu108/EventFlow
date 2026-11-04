"""Prerequisite checks and the bootstrap command that writes the skill."""

from pathlib import Path

import pytest

from tests.conftest import run_git
from tools import bootstrap_commit_skill as boot

FULL_IGNORE = "*.md\nplanning/\n.claude/\n.agents/\n.venv/\ndata/\n"


def test_ignore_gaps_lists_probe_paths_git_would_track(repo: Path) -> None:
    (repo / ".gitignore").write_text("*.md\n")

    gaps = boot.ignore_gaps(repo)

    assert "planning/notes.rst" in gaps
    assert ".claude/skills/commit/SKILL.md" not in gaps  # covered by *.md
    assert ".claude/settings.local.json" in gaps


def test_ignore_gaps_is_empty_with_the_full_rule_set(repo: Path) -> None:
    (repo / ".gitignore").write_text(FULL_IGNORE)

    assert boot.ignore_gaps(repo) == []


def test_tracked_violations_finds_force_added_markdown(repo: Path) -> None:
    (repo / "NOTES.md").write_text("x\n")
    run_git(repo, "add", "-f", "NOTES.md")

    assert boot.tracked_violations(repo) == ["NOTES.md"]


def test_tracked_violations_is_empty_for_a_clean_tree(repo: Path) -> None:
    assert boot.tracked_violations(repo) == []


def run_boot(repo: Path, reference: Path, skill: Path, *extra: str) -> int:
    policy = repo / "policy.toml"
    if not policy.exists():
        policy.write_text('auto_commit = true\ngranted_by = "m"\ntest_command = ["true"]\n')
    return boot.main(
        [
            "--repo", str(repo),
            "--policy", str(policy),
            "--reference", str(reference),
            "--skill-path", str(skill),
            *extra,
        ]
    )  # fmt: skip


@pytest.fixture
def reference(tmp_path: Path) -> Path:
    ref = tmp_path / "reference"
    (ref / ".agents" / "skills" / "commit").mkdir(parents=True)
    (ref / ".agents" / "skills" / "commit" / "SKILL.md").write_text("x\n")
    return ref


def test_bootstrap_writes_the_skill_and_reports_success(
    repo: Path, reference: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (repo / ".gitignore").write_text(FULL_IGNORE)
    skill = tmp_path / "out" / "SKILL.md"

    status = run_boot(repo, reference, skill)

    assert status == 0
    text = skill.read_text()
    assert "user.name  = Test Author" in text
    assert f"reviewed {reference}" in text
    assert "all prerequisites satisfied" in capsys.readouterr().out


def test_check_mode_writes_nothing(repo: Path, reference: Path, tmp_path: Path) -> None:
    (repo / ".gitignore").write_text(FULL_IGNORE)
    skill = tmp_path / "out" / "SKILL.md"

    assert run_boot(repo, reference, skill, "--check") == 0
    assert not skill.exists()


def test_unmet_prerequisites_exit_two_and_are_listed(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_git(repo, "config", "--local", "--unset", "user.email")
    skill = tmp_path / "out" / "SKILL.md"

    status = run_boot(repo, tmp_path / "absent", skill)

    out = capsys.readouterr().out
    assert status == 2
    assert "no commit-convention reference was reviewed" in out
    assert "git config --local user.name" in out
    assert "gitignore does not cover" in out
    assert "<UNSET -- ask the maintainer>" in skill.read_text()


def test_missing_template_exits_one(
    repo: Path, reference: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(boot, "TEMPLATE", tmp_path / "missing.txt")

    assert run_boot(repo, reference, tmp_path / "SKILL.md") == 1


def test_this_repository_satisfies_every_prerequisite_but_the_reference() -> None:
    assert boot.ignore_gaps(boot.REPO) == []
    assert boot.tracked_violations(boot.REPO) == []
