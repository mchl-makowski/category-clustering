"""Run logging and progress: `.txt` run log at DEBUG, console via `setup_logging` only, progress per long loop."""

import re

import pandas as pd
import pytest

from problem_clustering import compare, judge, judge_safely, load_config, load_tickets, run_method, setup_logging, write_outputs
from problem_clustering.backend import FakeBackend, content_filtered
from problem_clustering.cli import main

TXT_LINE = re.compile(r"\[\d\d:\d\d:\d\d\] (DEBUG|INFO|WARNING|ERROR) \S.*")
CONSOLE_LINE = re.compile(r"\d\d:\d\d:\d\d (DEBUG|INFO|WARNING|ERROR) \S.*")
BAR_NOISE = re.compile(r"\r|it/s|█")


@pytest.fixture
def tickets(fixture_path):
    return load_tickets(fixture_path)


def cfg(**over):
    return load_config(None, {"backend.kind": "fake", "method_a.distance_threshold": 0.6, **over})


def console_lines(err: str) -> list[str]:
    return [line for line in err.splitlines() if CONSOLE_LINE.fullmatch(line)]


# --- .txt run log -------------------------------------------------------------

def test_txt_log_records_debug_with_level_per_line_and_no_bar_frames(tickets, tmp_path):
    setup_logging(cfg(**{"logging.level": "WARNING", "logging.progress": "on"}))  # console level never limits .txt
    result = run_method(tickets, "c", cfg(), FakeBackend())
    assert all(TXT_LINE.fullmatch(line) for line in result.log), [l for l in result.log if not TXT_LINE.fullmatch(l)]
    assert any(" DEBUG config {" in line for line in result.log)
    assert any(re.search(r"DEBUG method_c: row \d+ -> ", line) for line in result.log)
    txt = write_outputs(result, None, tmp_path)["txt"].read_text(encoding="utf-8")
    assert not BAR_NOISE.search(txt)


def test_txt_log_has_final_count_per_loop_but_no_progress_lines(tickets):
    setup_logging(cfg(**{"logging.progress": "off"}))  # ~10% lines go to the console only
    result = run_method(tickets, "c", cfg(**{"method_c.consolidate_every": 5}), FakeBackend())
    log = "\n".join(result.log)
    n = len(tickets) - 2  # two empty Tickets (X-0000)
    assert not re.search(rf"INFO method_c Tickets: \d+/{n} problems=\d+$", log, re.M)  # progress noise
    assert re.search(rf"INFO method_c Tickets: {n}/{n} problems=\d+ done$", log, re.M)
    assert re.search(rf"INFO embeddings: {n}/{n} done$", log, re.M)  # counted in Tickets, not batches
    a = "\n".join(run_method(tickets, "a", cfg(), FakeBackend()).log)  # C names its own Problems; A doesn't
    assert re.search(r"INFO naming Problems: (\d+)/\1 done$", a, re.M)


def test_every_long_loop_reports_progress(tickets):
    b = "\n".join(run_method(tickets, "b", cfg(**{"method_b.coarse_threshold": 0.9}), FakeBackend()).log)
    assert re.search(r"INFO method_b iteration 1 verification: (\d+)/\1 done$", b, re.M)
    assert re.search(r"INFO method_b iteration 1 merge checks: (\d+)/\1 done$", b, re.M)
    s = run_method(tickets, "a", cfg(**{"symptom_summary.enabled": True}), FakeBackend())
    assert any(re.search(r"INFO symptom summary: 18/18 done$", line) for line in s.log)
    before = len(s.log)
    judge(s, cfg(), FakeBackend())
    assert any(re.search(r"INFO judge Split checks: (\d+)/\1 done$", line) for line in s.log[before:])


def test_judge_lines_are_appended_to_the_run_log(tickets):
    result = run_method(tickets, "a", cfg(), FakeBackend())
    before = len(result.log)
    judge(result, cfg(), FakeBackend())
    added = result.log[before:]
    assert any("judge Mixed checks:" in line and "done" in line for line in added)
    assert any("INFO judge: mixed_checks=" in line for line in added)
    assert all(TXT_LINE.fullmatch(line) for line in added)


def test_judge_safely_failure_is_a_warning_in_the_run_log(tickets, monkeypatch):
    from problem_clustering.backend.fake import DEFAULT_HANDLERS

    def broken(req):
        raise RuntimeError("boom")

    monkeypatch.setitem(DEFAULT_HANDLERS, "judge_mixed", broken)
    result = run_method(tickets, "a", cfg(), FakeBackend())
    before = len(result.log)
    assert judge_safely(result, cfg(), FakeBackend()) is None
    assert re.search(r"ERROR judge failed, run written without Judge: RuntimeError: boom", result.log[-1])
    assert not any("judge Mixed checks:" in line and "done" in line for line in result.log[before:])


def test_cli_judge_failure_shows_at_error_level(fixture_path, tmp_path, capsys, monkeypatch):
    from problem_clustering.backend.fake import DEFAULT_HANDLERS

    monkeypatch.setitem(DEFAULT_HANDLERS, "judge_mixed", lambda req: (_ for _ in ()).throw(RuntimeError("boom")))
    flags = ["--set", "logging.level=ERROR", "--no-progress"]
    assert main(["run", "--method", "a", "--input", str(fixture_path), "--out-dir", str(tmp_path),
                 "--set", "backend.kind=fake", "--set", "method_a.distance_threshold=0.6", *flags]) == 1
    assert " ERROR judge failed" in capsys.readouterr().err


def test_runs_do_not_leak_lines_into_each_other(tickets):
    first = run_method(tickets, "a", cfg(), FakeBackend())
    second = run_method(tickets, "a", cfg(), FakeBackend())
    assert len(first.log) == len(second.log)


# --- Problem counts -------------------------------------------------------------

@pytest.mark.parametrize("method", ["a", "b", "c"])
def test_final_problem_counts_logged_for_every_method(tickets, method):
    result = run_method(tickets, method, cfg(), FakeBackend())
    problems = len(result.catalogue) - 1  # minus X-0000
    singletons = int((result.catalogue["ticket_count"] == 1).sum())
    assert any(re.search(rf"INFO problems={problems} one-Ticket={singletons} X-0000=2$", l) for l in result.log)


def test_method_step_counts(tickets):
    a = "\n".join(run_method(tickets, "a", cfg(), FakeBackend()).log)
    assert re.search(r"INFO method_a: agglomerative .* -> \d+ Problems", a)
    b = "\n".join(run_method(tickets, "b", cfg(), FakeBackend()).log)
    assert re.search(r"INFO method_b: coarse threshold=.* -> \d+ candidate Problems", b)
    assert re.search(r"INFO method_b iteration 1: .*splits=\d+ merges=\d+ Problems=\d+", b)
    c = "\n".join(run_method(tickets, "c", cfg(**{"method_c.consolidate_every": 5}), FakeBackend()).log)
    assert re.search(r"INFO method_c: consolidation after 5 Tickets: .* \d+ remain", c)
    assert re.search(r"INFO method_c: .* -> \d+ Problems$", c, re.M)


# --- console -------------------------------------------------------------------

def test_no_console_output_without_setup_logging(tickets, capfd):
    fake = FakeBackend().on("problem_naming", content_filtered())  # WARNINGs too (no stdlib lastResort)
    result = run_method(tickets, "c", cfg(), fake)
    judge(result, cfg(), FakeBackend())
    out, err = capfd.readouterr()
    assert out == "" and err == ""


def test_console_level_and_format(tickets, capsys):
    setup_logging(cfg(**{"logging.progress": "off"}))
    run_method(tickets, "a", cfg(), FakeBackend())
    err = capsys.readouterr().err
    lines = console_lines(err)
    assert lines and len(lines) == len(err.splitlines())
    assert any(" INFO problems=" in line for line in lines)
    assert not any(" DEBUG " in line for line in lines)

    setup_logging(cfg(**{"logging.level": "debug", "logging.progress": "off"}))  # replaces the handler
    run_method(tickets, "a", cfg(), FakeBackend())
    lines = console_lines(capsys.readouterr().err)
    assert any(" DEBUG config " in line for line in lines)
    assert sum(" INFO problems=" in line for line in lines) == 1  # one handler, not two


def test_progress_off_logs_processed_total_lines_on_console(tickets, capsys):
    setup_logging(cfg(**{"logging.progress": "off"}))
    run_method(tickets, "c", cfg(**{"backend.embed_batch_size": 2}), FakeBackend())
    err = capsys.readouterr().err
    assert re.search(r"INFO method_c Tickets: \d+/18 problems=\d+$", err, re.M)
    assert re.search(r"INFO embeddings: \d+/18$", err, re.M)  # parallel batches, counted in Tickets
    assert not BAR_NOISE.search(err)


def test_progress_on_shows_a_bar_instead_of_progress_lines(tickets, capsys):
    setup_logging(cfg(**{"logging.progress": "on"}))
    result = run_method(tickets, "c", cfg(), FakeBackend())
    err = capsys.readouterr().err
    assert "method_c Tickets" in err and "\r" in err  # tqdm bar frames
    assert not re.search(r"INFO method_c Tickets: \d+/18 problems=\d+$", err, re.M)  # ~10% lines: .txt only
    assert re.search(r"INFO method_c Tickets: 18/18 problems=\d+ done", err)  # count line after the bar
    assert not any(re.search(r"method_c Tickets: \d+/18 problems=\d+$", l) for l in result.log)


def test_progress_auto_without_tty_has_no_bar(tickets, capsys):
    setup_logging(cfg())  # pytest's stderr is not a TTY and this is not Jupyter
    run_method(tickets, "a", cfg(), FakeBackend())
    assert not BAR_NOISE.search(capsys.readouterr().err)


def test_ticket_text_never_reaches_the_console(tickets, capsys):
    marker = "zzpodsumowaniezz"
    over = {"logging.level": "DEBUG", "logging.progress": "off", "symptom_summary.enabled": True,
            "symptom_summary.log_summaries": True}
    setup_logging(cfg(**over))
    fake = FakeBackend().on("symptom_summary", {"summary": marker}).on("problem_naming", content_filtered())
    result = run_method(tickets, "a", cfg(**over), fake)  # A: names come from raw text, not summaries
    err = capsys.readouterr().err
    assert marker in "\n".join(result.log)  # log_summaries: .txt only
    assert marker not in err
    for text in [*tickets["Opis"].dropna(), *tickets["Tytuł"].dropna()]:
        for chunk in re.split(r"[\n.]", str(text)):
            if len(chunk.strip()) >= 20:
                assert chunk.strip() not in err, chunk


def test_yaml_on_off_booleans_are_accepted(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("logging:\n  progress: off\n  level: warning\n", encoding="utf-8")
    config = load_config(path)
    assert (config.logging.progress, config.logging.level) == ("off", "WARNING")
    assert load_config(None, ["logging.progress=on"]).logging.progress == "on"


# --- CLI -----------------------------------------------------------------------

def _run_cli(fixture_path, tmp_path, *flags):
    return main(["run", "--method", "a", "--input", str(fixture_path), "--out-dir", str(tmp_path),
                 "--set", "backend.kind=fake", "--set", "method_a.distance_threshold=0.6", "--no-judge", *flags])


def test_cli_verbose_and_quiet(fixture_path, tmp_path, capsys):
    assert _run_cli(fixture_path, tmp_path, "-v", "--no-progress") == 0
    assert " DEBUG config " in capsys.readouterr().err
    assert _run_cli(fixture_path, tmp_path, "-q") == 0
    captured = capsys.readouterr()
    assert console_lines(captured.err) == []
    assert captured.out.count(".xlsx") == 1  # output paths still printed


def test_cli_verbose_and_quiet_are_mutually_exclusive(fixture_path, tmp_path):
    with pytest.raises(SystemExit):
        _run_cli(fixture_path, tmp_path, "-v", "-q")


def test_cli_flags_override_config_level(fixture_path, tmp_path, capsys):
    assert _run_cli(fixture_path, tmp_path, "--set", "logging.level=DEBUG", "-q") == 0
    assert console_lines(capsys.readouterr().err) == []


def test_cli_set_logging_level(fixture_path, tmp_path, capsys):
    assert _run_cli(fixture_path, tmp_path, "--set", "logging.level=DEBUG", "--no-progress") == 0
    assert " DEBUG config " in capsys.readouterr().err


def test_compare_logs_a_few_info_lines(tickets, tmp_path, capsys):
    paths = [write_outputs(run_method(tickets, m, cfg(), FakeBackend()), None, tmp_path)["xlsx"] for m in "ab"]
    assert main(["compare", *map(str, paths), "--out-dir", str(tmp_path)]) == 0
    lines = console_lines(capsys.readouterr().err)
    assert len(lines) == 3
    assert all(" INFO compare: " in line for line in lines)
    assert pd.read_excel(compare(paths, tmp_path), sheet_name="Comparison").shape[1] == 3
