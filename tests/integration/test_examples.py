"""The shipped examples must keep running."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("script", ["quickstart.py", "lobster_adapter.py"])
def test_example_runs(script: str, tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", [script, str(tmp_path / "out")])
    runpy.run_path(str(ROOT / "examples" / script), run_name="__main__")
    assert capsys.readouterr().out
