from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from eventflow.errors import ConfigError
from eventflow.reporting import to_csv, to_rst, write_table
from eventflow.reporting.tables import comparison_rows
from eventflow.training import environment_info, git_revision, resolve_device, seed_everything

ROWS = [{"run": "a", "model": "hybrid", "commit": "f" * 40, "event_type_tv": 0.123456, "x": None}]


def test_table_formats(tmp_path: Path) -> None:
    rst = to_rst(ROWS)
    assert rst.count("=====") >= 3 and "0.1235" in rst and "f" * 12 in rst and "f" * 13 not in rst
    assert to_csv(ROWS).splitlines()[0] == "run,model,commit,event_type_tv,x"
    for suffix in (".csv", ".json", ".rst"):
        assert write_table(ROWS, tmp_path / f"t{suffix}").read_text()
    assert json.loads((tmp_path / "t.json").read_text())[0]["run"] == "a"
    with pytest.raises(ConfigError):
        write_table(ROWS, tmp_path / "t.xlsx")
    assert to_rst([]) == "" and to_csv([]) == ""


def test_comparison_rows_leave_failed_metrics_empty() -> None:
    run = {
        "dir": "runs/x",
        "manifest": {"model_kind": "ddpm", "parameters": 5, "git": {"commit": None}},
        "evaluation": {
            "results": [
                {
                    "name": "event_type_marginal",
                    "scope": "all",
                    "status": "ok",
                    "values": {"tv": 0.2},
                },
                {"name": "side_marginal", "scope": "all", "status": "skipped", "values": {}},
            ]
        },
    }
    row = comparison_rows([run])[0]
    assert row["event_type_tv"] == 0.2 and row["side_tv"] is None and row["price_tv"] is None


def test_device_resolution_and_environment() -> None:
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("auto").type in ("cpu", "cuda", "mps")
    if not torch.cuda.is_available():
        with pytest.raises(ConfigError, match="CUDA"):
            resolve_device("cuda")
    info = environment_info(torch.device("cpu"))
    assert info["device"] == "cpu" and info["torch"] == torch.__version__


def test_git_revision_outside_a_repository(tmp_path: Path) -> None:
    assert git_revision(tmp_path) == {"commit": None, "dirty": None}
    assert git_revision(Path(__file__).parent)["commit"] is not None


def test_seed_everything_makes_torch_reproducible() -> None:
    seed_everything(5)
    a = torch.rand(3)
    seed_everything(5)
    assert torch.equal(a, torch.rand(3))
