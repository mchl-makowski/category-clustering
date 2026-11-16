"""CLI: argument parsing + public API calls only. No Method logic here.

    problem-clustering run --method a --input export.xlsx [--config config.yaml] [--set key=value ...]
                           [--out-dir outputs] [--no-judge] [-v | -q] [--no-progress]

    problem-clustering compare run1.xlsx run2.xlsx ... [--out-dir outputs] [-v | -q]

Console verbosity: `logging.level` (-v = DEBUG, -q = WARNING win over it), progress bars:
`logging.progress` (--no-progress = off). The `.txt` run log always records DEBUG.
"""

from __future__ import annotations

import argparse
import sys

from . import compare, judge_safely, load_config, load_tickets, run_method, setup_logging, write_outputs
from .backend import make_backend


def _logging_overrides(args: argparse.Namespace) -> list[str]:
    """CLI flags as `--set` overrides, applied last so they win over YAML and `--set`."""
    out = []
    if args.verbose:
        out.append("logging.level=DEBUG")
    if args.quiet:
        out.append("logging.level=WARNING")
    if getattr(args, "no_progress", False):
        out.append("logging.progress=off")
    return out


def _cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config, args.set + _logging_overrides(args))
    setup_logging(config)
    tickets = load_tickets(args.input)
    backend = make_backend(config.backend)
    result = run_method(tickets, args.method, config, backend)
    judge_result = None if args.no_judge else judge_safely(result, config, backend)
    paths = write_outputs(result, judge_result, args.out_dir)
    print(f"{paths['xlsx']}\n{paths['txt']}")
    if not args.no_judge and judge_result is None:
        return 1  # the Judge failure was logged (WARNING: console + .txt)
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    setup_logging(load_config(None, _logging_overrides(args)))
    print(compare(args.workbooks, args.out_dir))
    return 0


def _verbosity(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="store_true", help="Console at DEBUG (overrides logging.level)")
    group.add_argument("-q", "--quiet", action="store_true", help="Console at WARNING (overrides logging.level)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="problem-clustering")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run one Method on an Excel export")
    _verbosity(run)
    run.add_argument("--no-progress", action="store_true", help="No progress bars (= logging.progress=off)")
    run.add_argument("--method", required=True, choices=["a", "b", "c"])
    run.add_argument("--input", required=True, help="Service Desk export (.xlsx)")
    run.add_argument("--config", default=None, help="YAML config")
    run.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                     help="Override a config value, dotted key (repeatable)")
    run.add_argument("--out-dir", default="outputs")
    run.add_argument("--no-judge", action="store_true", help="Skip the Judge")
    run.set_defaults(func=_cmd_run)

    cmp = sub.add_parser("compare", help="Put the Metrics of several run workbooks side by side")
    _verbosity(cmp)
    cmp.add_argument("workbooks", nargs="+", help="Run workbooks (.xlsx) written by `run`")
    cmp.add_argument("--out-dir", default="outputs")
    cmp.set_defaults(func=_cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
