"""Command-line interface: ``eventflow <command> ...``.

Commands
--------
``validate CONFIG...``      parse and validate configurations
``compare CONFIG...``       report differences that would make a comparison unfair
``dry-run CONFIG``          initialise everything, one CPU forward/backward, no training
``synth OUT.csv``           write a synthetic event dataset
``inspect DATA.csv``        validate a CSV event table and print its summary
``train CONFIG``            full train -> generate -> evaluate pipeline (needs ``--yes``)
``report RUN_DIR...``       tabulate evaluation results of finished runs

``train`` launches real optimisation; it refuses to start without ``--yes``
so that a dry run is never confused with an experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from eventflow.errors import EventFlowError


def _print(data: object) -> None:
    print(json.dumps(data, indent=2, sort_keys=True, default=str))


def cmd_validate(args: argparse.Namespace) -> int:
    from eventflow.experiments.config import load_config

    for path in args.configs:
        cfg = load_config(path)
        print(
            f"ok  {path}  name={cfg.name} model={cfg.model_kind} "
            f"token_budget={cfg.training_token_budget()} "
            f"nfe_per_event={cfg.sampling_evaluations_per_event()}"
        )
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from eventflow.experiments.config import comparability_issues, load_config

    configs = [load_config(p) for p in args.configs]
    counts = None
    if args.parameters:
        from eventflow.experiments.runner import build_model, prepare

        prepared = prepare(configs[0])
        counts = {
            c.name: build_model(c, prepared).num_parameters() for c in configs if not c.is_baseline
        }
        _print({"parameters": counts})
    issues = comparability_issues(configs, counts, args.tolerance)
    for issue in issues:
        print(f"UNFAIR  {issue}")
    if not issues:
        print(f"comparable: {', '.join(c.name for c in configs)}")
    return 1 if issues else 0


def cmd_dry_run(args: argparse.Namespace) -> int:
    from eventflow.experiments.config import load_config
    from eventflow.experiments.runner import dry_run

    report = dry_run(load_config(args.config))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    _print(report)
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    from eventflow.marketdata.io import write_csv
    from eventflow.marketdata.synthetic import SyntheticConfig, generate_synthetic

    frame = generate_synthetic(
        SyntheticConfig(
            seed=args.seed, sessions_per_instrument=args.sessions, events_per_session=args.events
        )
    )
    write_csv(frame, args.output)
    _print(frame.summary())
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    from eventflow.marketdata.io import read_csv
    from eventflow.marketdata.preprocess import fingerprint
    from eventflow.marketdata.schema import validate_frame

    frame = read_csv(args.path, message_level=args.message_level)
    _print(
        {
            **frame.summary(),
            "fingerprint": fingerprint(frame),
            "validation": validate_frame(frame).to_dict(),
        }
    )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from eventflow.experiments.config import load_config
    from eventflow.experiments.runner import run_experiment

    if not args.yes:
        print("refusing to train without --yes (run `eventflow dry-run` first)", file=sys.stderr)
        return 2
    cfg = load_config(args.config)
    _print(run_experiment(cfg, Path(args.run_dir) if args.run_dir else None, resume=args.resume))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from eventflow.reporting.tables import collect_runs, comparison_rows, write_table

    runs = collect_runs([Path(p) for p in args.runs])
    rows = comparison_rows(runs)
    if args.output:
        write_table(rows, Path(args.output))
    else:
        from eventflow.reporting.tables import to_rst

        print(to_rst(rows))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eventflow", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="validate experiment configs")
    p.add_argument("configs", nargs="+")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("compare", help="check that experiment configs are comparable")
    p.add_argument("configs", nargs="+")
    p.add_argument(
        "--parameters",
        action="store_true",
        help="also build every model and compare parameter counts",
    )
    p.add_argument("--tolerance", type=float, default=0.25)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("dry-run", help="initialise all components without training")
    p.add_argument("config")
    p.add_argument("--output", help="also write the report to this JSON file")
    p.set_defaults(func=cmd_dry_run)

    p = sub.add_parser("synth", help="write a synthetic event CSV")
    p.add_argument("output")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sessions", type=int, default=2, help="sessions per instrument")
    p.add_argument("--events", type=int, default=2000, help="events per session")
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("inspect", help="validate and summarise an event CSV")
    p.add_argument("path")
    p.add_argument(
        "--message-level",
        action="store_true",
        help="declare that the table holds order-book messages",
    )
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("train", help="run the full pipeline (real training)")
    p.add_argument("config")
    p.add_argument("--run-dir")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--yes", action="store_true", help="confirm that training should start")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("report", help="tabulate finished runs")
    p.add_argument("runs", nargs="+")
    p.add_argument("--output", help=".csv, .json or .rst output path")
    p.set_defaults(func=cmd_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except EventFlowError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
