"""CLI: argument parsing + public API calls only. No Method logic here.

    problem-clustering run --method a --input export.xlsx [--config config.yaml] [--set key=value ...]
                           [--out-dir outputs] [--no-judge]

`compare` is added by #7 as another subparser.
"""

from __future__ import annotations

import argparse
import sys

from . import load_config, load_tickets, run_method, write_outputs
from .backend import make_backend


def _cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config, args.set)
    tickets = load_tickets(args.input)
    backend = make_backend(config.backend)
    result = run_method(tickets, args.method, config, backend)
    judge_result = None  # Judge wired in by #7 unless --no-judge
    paths = write_outputs(result, judge_result, args.out_dir)
    print(f"{paths['xlsx']}\n{paths['txt']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="problem-clustering")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run one Method on an Excel export")
    run.add_argument("--method", required=True, choices=["a", "b", "c"])
    run.add_argument("--input", required=True, help="Service Desk export (.xlsx)")
    run.add_argument("--config", default=None, help="YAML config")
    run.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                     help="Override a config value, dotted key (repeatable)")
    run.add_argument("--out-dir", default="outputs")
    run.add_argument("--no-judge", action="store_true", help="Skip the Judge")
    run.set_defaults(func=_cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
