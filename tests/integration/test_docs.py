"""Public documentation must be valid reStructuredText and mention no Markdown files."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", ["README.rst", "CONTRIBUTING.rst"])
def test_rst_parses_without_warnings(name: str) -> None:
    docutils_core = pytest.importorskip("docutils.core")
    text = (ROOT / name).read_text()
    docutils_core.publish_doctree(text, settings_overrides={"report_level": 2, "halt_level": 2})
    assert ".md" not in text


def test_readme_documented_commands_exist() -> None:
    from eventflow.cli import build_parser

    commands = set(build_parser()._subparsers._group_actions[0].choices)  # type: ignore[union-attr]
    readme = (ROOT / "README.rst").read_text()
    for command in ("validate", "compare", "dry-run", "train", "report"):
        assert f"eventflow {command}" in readme and command in commands
