# Run logging and progress reporting with loguru

## Goal

A run currently prints only the output file paths. While it runs (minutes for ~2500 Tickets with Method C or the Judge) there is no sign of progress. We want live feedback on the console: how many Tickets have been processed, ideally as a progress bar, and how many Problems (clusters) each step produces. How much is printed must be configurable.

## Current state

- `run.py` collects timestamped lines in `RunResult.log` via `log_line`, passed to Methods as `ctx.log`. `output.py` writes them to `yymmdd_hhmm_<method>.txt`.
- Nothing reaches the console until the run ends. There is no logging library, no progress bar and no verbosity setting.

## Requirements

1. **Use loguru** for all run logging. Replace the ad-hoc `log_line` / `ctx.log` callables with loguru calls (or route them through loguru). Add `loguru` as a dependency.
2. **Progress for long loops.** Show processed / total items (Tickets or Problems) for every long loop, as a progress bar where the console supports it:
    - embeddings,
    - Symptom Summary (when enabled),
    - Method B LLM verification and merge checks,
    - Method C agent, Ticket by Ticket,
    - Problem naming (per Problem),
    - Judge checks (per Problem / pair).
3. **Cluster counts.** Log the number of Problems after each step that changes it:
    - Method A: Problems after clustering,
    - Method B: candidates after coarse clustering, then per iteration (splits, merges, current count),
    - Method C: current catalogue size at each consolidation, and the final count,
    - every Method: final Problem count, one-Ticket Problems, `X-0000` size.
4. **Configurable verbosity.**
    - A config key for the console level (e.g. `logging.level: INFO`), settable in YAML and with `--set logging.level=DEBUG`.
    - CLI shortcuts `-v` / `-q` (more / less output) are welcome.
    - A switch to turn the progress bar off (non-interactive Foundry jobs, CI).
5. **Keep the `.txt` run log.** It still gets written next to the workbook and contains at least what it has today. Progress bar frames do not go into it.
6. **Notebook.** Logging and progress work in Jupyter too (progress bar renders in the notebook, or is cleanly disabled).

## Constraints

- Logging lives in the shared module. CLI and notebook only configure it (level, progress on/off).
- Log messages must not include Ticket text beyond what the current log already allows (`symptom_summary.log_summaries`). PII stays out of the console.
- Tests stay offline and quiet: pytest output is not flooded with progress bars.
- New config section and keys are documented in `README.md` (`tests/test_readme.py` enforces).

## Acceptance criteria

- [ ] `run` shows a progress bar (or periodic processed/total lines) for every long loop listed above.
- [ ] Problem counts are logged after each step listed above, for all three Methods.
- [ ] Console verbosity is set from config and CLI; progress bar can be turned off.
- [ ] The `.txt` run log is still written and has no progress bar noise.
- [ ] The notebook runs without broken progress output.
- [ ] README documents the new config keys and flags; tests pass.

## Open questions

- Progress bar library: loguru plus `tqdm` (with `tqdm.auto` for Jupyter), or `rich`?
- Default console level: `INFO` (steps + counts) with `DEBUG` for per-Ticket detail?
- Should the `.txt` log follow the console level or always record at `DEBUG`?
