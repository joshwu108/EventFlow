"""End-to-end CPU checks on the tiny smoke configuration (seconds, not an experiment)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import torch

from eventflow.cli import main
from eventflow.errors import ConfigError
from eventflow.experiments import comparability_issues, config_from_dict, load_config
from eventflow.experiments.config import BaselineConfig, read_config_table
from eventflow.experiments.runner import (
    dry_run,
    load_model_from_checkpoint,
    prepare,
    run_experiment,
)
from eventflow.models.config import ModelConfig
from eventflow.reporting import collect_runs, comparison_rows, to_rst
from eventflow.tokenization import collate_events

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "configs" / "smoke.toml"
RESEARCH = ["baseline", "transformer", "ddpm", "cfm", "hybrid"]


@pytest.fixture(scope="module")
def smoke():
    return load_config(SMOKE)


def _with_model(cfg, kind: str):
    if kind == "baseline":
        return dataclasses.replace(cfg, name=f"smoke_{kind}", model=BaselineConfig())
    fields = {f.name: getattr(cfg.model, f.name) for f in dataclasses.fields(ModelConfig)}
    fields.pop("kind"), fields.pop("categorical_mode"), fields.pop("continuous_head")
    return dataclasses.replace(cfg, name=f"smoke_{kind}", model=ModelConfig.preset(kind, **fields))


@pytest.mark.parametrize("kind", RESEARCH)
def test_dry_run_initialises_every_model_family(smoke, kind) -> None:
    report = dry_run(_with_model(smoke, kind))
    assert report["windows"]["train"] > 0
    assert report["parameters"] > 0
    assert set(report["fingerprints"]) == {"source", "train", "validation", "test"}
    if kind != "baseline":
        assert report["initial_loss"]["total"] > 0
    json.dumps(report, default=str)


def test_all_research_configs_validate_and_are_comparable() -> None:
    configs = [load_config(ROOT / "configs" / f"{name}.toml") for name in RESEARCH]
    assert comparability_issues(configs) == []
    altered = dataclasses.replace(configs[2], train=dataclasses.replace(configs[2].train, lr=1e-3))
    assert any("train.lr" in issue for issue in comparability_issues([configs[1], altered]))
    shorter = dataclasses.replace(
        configs[4], sequence=dataclasses.replace(configs[4].sequence, context_length=64)
    )
    assert any("[sequence]" in issue for issue in comparability_issues([configs[1], shorter]))


def test_parameter_budgets_are_flagged(smoke) -> None:
    configs = [load_config(ROOT / "configs" / f"{n}.toml") for n in ("transformer", "hybrid")]
    issues = comparability_issues(configs, {"transformer": 100, "hybrid": 1000})
    assert any("parameters" in issue for issue in issues)


def test_config_errors_are_specific(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown keys"):
        config_from_dict({"experiment": {"name": "x"}, "train": {"lr": 1e-3, "lrr": 1}})
    with pytest.raises(ConfigError, match="max_context"):
        config_from_dict(
            {
                "experiment": {"name": "x"},
                "sequence": {"context_length": 512},
                "model": {"kind": "hybrid", "max_context": 256},
            }
        )
    loop = tmp_path / "a.toml"
    loop.write_text('[experiment]\nname = "a"\nextends = "a.toml"\n')
    with pytest.raises(ConfigError, match="circular"):
        read_config_table(loop)


def test_full_pipeline_trains_generates_evaluates_and_reports(smoke, tmp_path: Path) -> None:
    runs = []
    for kind in ("baseline", "hybrid"):
        cfg = _with_model(smoke, kind)
        run_dir = tmp_path / cfg.name
        summary = run_experiment(cfg, run_dir)
        assert summary["training"]["status"] in ("completed", "fitted")
        assert 0.0 <= summary["rollout_invalid_rate"] <= 1.0
        evaluation = json.loads((run_dir / "evaluation.json").read_text())
        categories = {r["category"] for r in evaluation["results"]}
        assert {"marginal", "conditional", "temporal", "structural"} <= categories
        runs.append(run_dir)
    rows = comparison_rows(collect_runs(runs))
    assert [r["model"] for r in rows] == ["baseline", "hybrid"]
    assert "event_type_tv" in to_rst(rows)

    model, tokenizer, payload = load_model_from_checkpoint(runs[1] / "checkpoints" / "final.pt")
    prepared = prepare(_with_model(smoke, "hybrid"))
    assert tokenizer.layout == prepared.tokenizer.layout
    test_windows = prepared.datasets["test"]
    batch = collate_events([test_windows[0], test_windows[1]])
    assert torch.isfinite(model.loss(batch).total)
    assert payload["step"] == smoke.train.max_steps


def test_reporting_refuses_untraceable_runs(tmp_path: Path) -> None:
    (tmp_path / "evaluation.json").write_text("{}")
    with pytest.raises(ConfigError, match="manifest"):
        collect_runs([tmp_path])


def test_cli_commands(tmp_path: Path, capsys) -> None:
    assert main(["validate", str(SMOKE)]) == 0
    csv_path = tmp_path / "synthetic.csv"
    assert main(["synth", str(csv_path), "--sessions", "1", "--events", "50"]) == 0
    assert main(["inspect", str(csv_path)]) == 0
    assert '"fingerprint"' in capsys.readouterr().out
    assert main(["dry-run", str(SMOKE), "--output", str(tmp_path / "dry.json")]) == 0
    assert json.loads((tmp_path / "dry.json").read_text())["experiment"] == "smoke"
    assert main(["train", str(SMOKE)]) == 2  # refuses without --yes
    research = [str(ROOT / "configs" / f"{n}.toml") for n in RESEARCH]
    assert main(["compare", *research]) == 0
    assert main(["validate", str(tmp_path / "missing.toml")]) == 1
