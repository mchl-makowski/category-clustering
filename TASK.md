# Run logging and progress reporting

## Goal

A run currently prints only the output file paths. While it runs (minutes for ~2500 Tickets with Method C or the Judge) there is no sign of progress. We want live feedback on the console: how many Tickets have been processed, ideally as a progress bar, and how many Problems (clusters) each step produces. How much is printed must be configurable.

## Current state

- `run.py` collects timestamped lines in `RunResult.log` via `log_line`, passed to Methods as `ctx.log`. `output.py` writes them to `yymmdd_hhmm_<method>.txt`.
- Nothing reaches the console until the run ends. There is no logging library, no progress bar and no verbosity setting.

## Requirements

1. **Use stdlib `logging`** for all run logging, with module loggers (`logging.getLogger(__name__)`) under the `problem_clustering` package logger. Remove `ctx.log`, `log_line` and the `log` callable parameters (`naming`, `summary`, `backend/retry`, the manual append in `judge_safely`). Add `tqdm` and `ipywidgets` as dependencies.
2. **Progress for long loops.** Show processed / total items (Tickets or Problems) for every long loop:
    - embeddings,
    - Symptom Summary (when enabled),
    - Method B LLM verification and merge checks,
    - Method C agent, Ticket by Ticket (bar postfix: current catalogue size),
    - Problem naming (per Problem),
    - Judge checks (per Problem / pair).
3. **Cluster counts.** Log the number of Problems at INFO after each step that changes it:
    - Method A: Problems after clustering,
    - Method B: candidates after coarse clustering, then per iteration (splits, merges, current count),
    - Method C: current catalogue size at each consolidation, and the final count,
    - every Method: final Problem count, one-Ticket Problems, `X-0000` size.
4. **Configurable verbosity.** New config section `logging`:
    - `level` (default `INFO`): console level. Settable in YAML and with `--set logging.level=DEBUG`.
    - `progress: auto | on | off` (default `auto`): `auto` shows a bar when stderr is a TTY or in Jupyter, otherwise none (Foundry jobs, CI).
    - CLI: `-v` (DEBUG) and `-q` (WARNING), mutually exclusive, override `logging.level`. `--no-progress` is the same as `logging.progress=off`.
5. **Keep the `.txt` run log.** It is still written next to the workbook and has at least what it has today. It always records at DEBUG, whatever the console level, and never contains progress bar frames.
6. **Notebook.** Calls `setup_logging(config)`. The progress bar renders as a `tqdm.auto` widget.
7. **`compare`.** A few INFO lines (workbooks read, output written). No bar, no per-row noise.

## Decided during grilling (2026-11-16)

- **Library:** stdlib `logging` + `tqdm` (`tqdm.auto`, `ipywidgets` for the notebook widget). Not loguru, not rich.
- **Console setup:** a new public `setup_logging(config)` in `__init__.py`, also added to `PUBLIC_API` in `tests/test_notebook.py`. The CLI and the notebook call it. `run_method` and `judge` never touch console handlers. Without `setup_logging` the package logger has only a `NullHandler`, so there is no console output and no bars, and tests stay quiet.
- **Levels:**
    - INFO: steps and Problem counts.
    - DEBUG: per-Ticket and per-Problem detail (Method C rows, merges, the config JSON).
    - WARNING: retries, content-filter fallbacks, Judge failure.
- **`.txt` capture:** a per-run in-memory handler is attached to the `problem_clustering` logger for the duration of `run_method` (and of `judge` / `judge_safely`, which appends to the same `RunResult.log`). It is removed afterwards. `write_outputs` is unchanged.
- **Line format:**
    - `.txt`: `[HH:MM:SS] LEVEL message`. The level is new, so update tests that match the old format.
    - Console: `HH:MM:SS LEVEL message`, written through `tqdm.write` so bars don't break.
- **Progress hook points:**
    - `parallel_map` takes an optional `desc=` and drives the bar itself. This covers naming, Symptom Summary, Judge, and Method B verification and merges.
    - Sequential loops (Method C, embedding batches) use a shared `progress(iterable, total, desc)` helper.
    - Bars are transient (`leave=False`), and an INFO count line follows each one.
- **No bar** (`off`, or `auto` without a TTY): a processed/total INFO line about every 10%, plus a final line per loop.
- **PII on the console:**
    - Allowed: row numbers, Problem ids, Problem names, Confidence Levels, counts.
    - Never: Ticket text.
    - `symptom_summary.log_summaries` lines go to the `.txt` only, even at DEBUG.

## Constraints

- Logging lives in the shared module. CLI and notebook only configure it (`setup_logging`).
- Log messages follow the PII rule above.
- Tests stay offline and quiet: pytest output is not flooded with progress bars.
- New config section, keys and CLI flags are documented in `README.md` (`tests/test_readme.py` enforces).
- After adding dependencies, re-export `requirements.txt`.

## Acceptance criteria

- [ ] `run` shows a progress bar (or ~10% processed/total lines) for every long loop listed above.
- [ ] Problem counts are logged after each step listed above, for all three Methods.
- [ ] Console verbosity is set from config and CLI (`-v` / `-q`); progress can be turned off (`logging.progress`, `--no-progress`).
- [ ] The `.txt` run log is still written, records at DEBUG, carries the level per line, and has no progress bar noise.
- [ ] The notebook runs with a working progress widget; `setup_logging` is part of the public API.
- [ ] No console output without `setup_logging`; Ticket text never reaches the console.
- [ ] README documents the new config keys and flags; `requirements.txt` updated; tests pass.
