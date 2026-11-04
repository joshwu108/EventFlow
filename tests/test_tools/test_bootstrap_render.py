"""The skill file is rendered from the policy, the identity and a reference review."""

from pathlib import Path

from tools.bootstrap_commit_skill import REFERENCE_CANDIDATES, render_skill, review_reference
from tools.policy import Policy

POLICY = Policy(
    auto_commit=True,
    auto_push=True,
    granted_on="2026-11-04",
    granted_by="maintainer",
    test_command=("python", "-m", "pytest", "-q"),
    lint_command=("ruff", "check", "."),
    requires_explicit_approval=("merge", "force-push"),
)


def test_render_fills_every_placeholder(tmp_path: Path) -> None:
    template = tmp_path / "t.txt"
    template.write_text(
        "{REFERENCE_STATUS}|{AUTHORIZATION}|{GIT_USER_NAME}|{GIT_USER_EMAIL}|"
        "{TEST_COMMAND}|{LINT_COMMAND}|{REMOTE_BRANCH}"
    )

    rendered = render_skill(template, POLICY, ("Name", "name@example.invalid"), "reviewed X")

    assert rendered == (
        "reviewed X|routine files may be committed without re-asking and tested milestones "
        "are pushed to origin/main (granted 2026-11-04 by maintainer). Always requires "
        "separate approval: merge, force-push.|Name|name@example.invalid|"
        "python -m pytest -q|ruff check .|origin/main"
    )


def test_render_marks_an_unset_identity_instead_of_inventing_one(tmp_path: Path) -> None:
    template = tmp_path / "t.txt"
    template.write_text("{GIT_USER_NAME} {GIT_USER_EMAIL}")

    rendered = render_skill(template, POLICY, ("", ""), "x")

    assert rendered == "<UNSET -- ask the maintainer> <UNSET -- ask the maintainer>"


def test_render_leaves_no_placeholder_behind_in_the_real_template() -> None:
    from tools.bootstrap_commit_skill import TEMPLATE

    rendered = render_skill(TEMPLATE, POLICY, ("N", "e@x"), "reviewed")

    assert "{" not in rendered
    assert rendered.startswith("---\nname: commit\n")


def test_review_reference_reports_a_missing_directory(tmp_path: Path) -> None:
    status, found = review_reference(tmp_path / "absent")

    assert status.startswith("NOT FOUND at ")
    assert found == []


def test_review_reference_reports_a_directory_without_a_skill(tmp_path: Path) -> None:
    status, found = review_reference(tmp_path)

    assert "no recognizable commit skill" in status
    assert found == []


def test_review_reference_lists_only_the_known_candidate_files(tmp_path: Path) -> None:
    for name in (".claude/skills/commit/SKILL.md", "CLAUDE.md", "secrets/.env"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x\n")

    status, found = review_reference(tmp_path)

    assert status == f"reviewed {tmp_path} -- .claude/skills/commit/SKILL.md, CLAUDE.md"
    assert [p.relative_to(tmp_path).as_posix() for p in found] == [
        ".claude/skills/commit/SKILL.md",
        "CLAUDE.md",
    ]
    assert all(not candidate.startswith("secrets") for candidate in REFERENCE_CANDIDATES)
