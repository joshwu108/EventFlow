"""Paths the commit tooling refuses to stage, whatever the caller asks for."""

from pathlib import Path

import pytest

from tools.paths import MAX_FIXTURE_BYTES, forbidden_reason


@pytest.mark.parametrize(
    "path",
    [
        "README.md",
        "docs/guide.MD",
        "notes.markdown",
        "src/pkg/story.mdx",
        "planning/roadmap.rst",
        "src/planning/notes.txt",
        ".planning/x.toml",
        "plans/q4.json",
        "docs/planning/a.rst",
        "docs/plans/b.rst",
        "feature_PRD.rst",
        "spec.prd",
        "spec.plan",
        ".claude/settings.local.json",
        ".agents/skills/commit/SKILL.md",
        ".venv/bin/python",
        "venv/lib/x.py",
        ".env",
        ".env.local",
        "config/.env.production",
        "keys/server.pem",
    ],
)
def test_private_and_secret_paths_are_always_forbidden(tmp_path: Path, path: str) -> None:
    assert forbidden_reason(tmp_path, path) is not None


@pytest.mark.parametrize(
    "path", ["data/trades.csv", "archive.zip", "weights.npy", "frame.parquet", "day.csv.gz"]
)
def test_datasets_are_forbidden_outside_fixtures(tmp_path: Path, path: str) -> None:
    assert forbidden_reason(tmp_path, path) is not None


@pytest.mark.parametrize(
    "path",
    [
        "tools/paths.py",
        "tests/test_tools/test_paths.py",
        "pyproject.toml",
        ".gitignore",
        "README.rst",
        ".github/workflows/ci.yml",
        "tools/commit_policy.toml",
        "results/run-1.json",
    ],
)
def test_source_tests_config_and_public_docs_are_allowed(tmp_path: Path, path: str) -> None:
    assert forbidden_reason(tmp_path, path) is None


def test_small_fixture_dataset_is_allowed(tmp_path: Path) -> None:
    fixture = tmp_path / "tests" / "fixtures" / "sample.csv.gz"
    fixture.parent.mkdir(parents=True)
    fixture.write_bytes(b"x" * 100)

    assert forbidden_reason(tmp_path, "tests/fixtures/sample.csv.gz") is None


def test_oversized_fixture_is_refused_with_its_size(tmp_path: Path) -> None:
    fixture = tmp_path / "tests" / "fixtures" / "huge.npy"
    fixture.parent.mkdir(parents=True)
    fixture.write_bytes(b"x" * (MAX_FIXTURE_BYTES + 1))

    reason = forbidden_reason(tmp_path, "tests/fixtures/huge.npy")

    assert reason is not None
    assert str(MAX_FIXTURE_BYTES + 1) in reason


def test_markdown_fixture_is_still_forbidden(tmp_path: Path) -> None:
    fixture = tmp_path / "tests" / "fixtures" / "README.md"
    fixture.parent.mkdir(parents=True)
    fixture.write_text("x\n")

    assert forbidden_reason(tmp_path, "tests/fixtures/README.md") is not None


def test_reason_names_the_matching_rule(tmp_path: Path) -> None:
    assert forbidden_reason(tmp_path, "planning/a.rst") == "directory 'planning'"
    assert forbidden_reason(tmp_path, "x/y/notes.md") == "pattern '*.md'"
