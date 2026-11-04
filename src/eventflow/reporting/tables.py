"""Comparison tables built only from files that finished runs wrote.

Nothing here computes or estimates a result: a table cell is either a value
read from a run's ``evaluation.json`` / ``summary.json`` / ``manifest.json``
or empty. Runs whose manifest is missing are rejected, so every row can be
traced to a configuration, dataset fingerprint and code revision.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from eventflow.errors import ConfigError

# (column, metric name, path into the metric's values)
HEADLINE_METRICS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("event_type_tv", "event_type_marginal", ("tv",)),
    ("side_tv", "side_marginal", ("tv",)),
    ("price_tv", "price_change_marginal", ("tv_ticks",)),
    ("log_qty_w1", "quantity_marginal", ("log_quantity", "wasserstein1")),
    ("log_gap_w1", "interarrival_marginal", ("log_positive_gap", "wasserstein1")),
    ("transition_tv", "type_transitions", ("weighted_tv",)),
    ("return_acf_diff", "return_autocorrelation", ("max_abs_diff",)),
    ("vol_cluster_diff", "volatility_clustering", ("max_abs_diff",)),
    ("invalid_rate", "structural_validity", ("invalid_event_rate",)),
    ("tstr_gap", "tstr_direction", ("accuracy_gap",)),
)


def collect_runs(run_dirs: Sequence[Path]) -> list[dict[str, Any]]:
    runs = []
    for run_dir in run_dirs:
        manifest = run_dir / "manifest.json"
        evaluation = run_dir / "evaluation.json"
        if not manifest.is_file():
            raise ConfigError(f"{run_dir}: no manifest.json; refusing to report an untraceable run")
        if not evaluation.is_file():
            raise ConfigError(f"{run_dir}: no evaluation.json (run not finished?)")
        runs.append(
            {
                "dir": str(run_dir),
                "manifest": json.loads(manifest.read_text()),
                "evaluation": json.loads(evaluation.read_text()),
                "summary": json.loads((run_dir / "summary.json").read_text())
                if (run_dir / "summary.json").is_file()
                else {},
            }
        )
    return runs


def _lookup(values: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = values
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def comparison_rows(runs: Sequence[dict[str, Any]], scope: str = "all") -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        manifest = run["manifest"]
        results = {(r["name"], r["scope"]): r for r in run["evaluation"]["results"]}
        row: dict[str, Any] = {
            "run": Path(run["dir"]).name,
            "model": manifest.get("model_kind"),
            "parameters": manifest.get("parameters"),
            "commit": (manifest.get("git") or {}).get("commit"),
            "test_fingerprint": (manifest.get("fingerprints") or {}).get("test"),
        }
        for column, metric, path in HEADLINE_METRICS:
            result = results.get((metric, scope))
            row[column] = (
                _lookup(result["values"], path) if result and result["status"] == "ok" else None
            )
        rows.append(row)
    return rows


def to_csv(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    text = str(value)
    return text[:12] if len(text) == 40 or len(text) == 64 else text  # abbreviate hashes


def to_rst(rows: Sequence[dict[str, Any]]) -> str:
    """reStructuredText simple table."""
    if not rows:
        return ""
    columns = list(rows[0])
    cells = [[_fmt(r[c]) for c in columns] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(columns)]
    rule = "  ".join("=" * w for w in widths)
    lines = [rule, "  ".join(c.ljust(w) for c, w in zip(columns, widths, strict=True)), rule]
    lines += ["  ".join(v.ljust(w) for v, w in zip(row, widths, strict=True)) for row in cells]
    lines.append(rule)
    return "\n".join(lines) + "\n"


def write_table(rows: Sequence[dict[str, Any]], path: Path) -> Path:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        path.write_text(to_csv(rows))
    elif suffix == ".json":
        path.write_text(json.dumps(list(rows), indent=2, default=str))
    elif suffix == ".rst":
        path.write_text(to_rst(rows))
    else:
        raise ConfigError(f"unsupported table format {suffix!r}; use .csv, .json or .rst")
    return path
