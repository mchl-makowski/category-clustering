"""Run logging and progress (stdlib `logging` + `tqdm.auto`).

Every module logs to `logging.getLogger(__name__)`, under the `problem_clustering`
package logger. That logger is always at DEBUG and has only a `NullHandler`, so
nothing reaches the console until `setup_logging(config)` (CLI, notebook) adds the
console handler. `run_method` / `judge` collect the `.txt` run log with `capture`.

Record tags (`extra=`):
- `TXT_ONLY`: `.txt` only, never the console (e.g. Symptom Summary text).
- `PROGRESS`: the ~10% processed/total lines; console only, and only when bars are off.

Never log Ticket text without `TXT_ONLY`.
"""

from __future__ import annotations

import logging
import sys
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterable, Iterator, TypeVar

from tqdm.auto import tqdm

if TYPE_CHECKING:
    from .config import Config

PACKAGE = "problem_clustering"
TXT_ONLY = {"txt_only": True}
PROGRESS = {"progress": True}
TXT_FORMAT = "[%(asctime)s] %(levelname)s %(message)s"
CONSOLE_FORMAT = "%(asctime)s %(levelname)s %(message)s"
DATE_FORMAT = "%H:%M:%S"
PROGRESS_STEPS = 10  # no bar: one processed/total line about every 10%

T = TypeVar("T")

_package = logging.getLogger(PACKAGE)
_package.setLevel(logging.DEBUG)  # the .txt always records DEBUG; the console handler filters by level
if not any(isinstance(h, logging.NullHandler) for h in _package.handlers):
    _package.addHandler(logging.NullHandler())  # no stdlib lastResort output without setup_logging
log = logging.getLogger(__name__)


class _ConsoleHandler(logging.Handler):
    """Writes through `tqdm.write` (to the current `sys.stderr`) so bars aren't broken."""

    def __init__(self, level: int, bars: bool):
        super().__init__(level)
        self.bars = bars
        self.setFormatter(logging.Formatter(CONSOLE_FORMAT, DATE_FORMAT))

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "txt_only", False):
            return False
        if self.bars and getattr(record, "progress", False):
            return False
        return super().filter(record)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=sys.stderr)
        except Exception:  # noqa: BLE001 - stdlib handler contract
            self.handleError(record)


class _ListHandler(logging.Handler):
    def __init__(self, lines: list[str]):
        super().__init__(logging.DEBUG)
        self.lines = lines
        self.setFormatter(logging.Formatter(TXT_FORMAT, DATE_FORMAT))

    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(record, "progress", False) and super().filter(record)  # no progress noise in .txt

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


def setup_logging(config: Config) -> None:
    """Console output per `config.logging` (level, progress bars). Safe to call again: replaces the handler."""
    cfg = config.logging
    for h in [h for h in _package.handlers if isinstance(h, _ConsoleHandler)]:
        _package.removeHandler(h)
    bars = cfg.progress == "on" or (cfg.progress == "auto" and (_stderr_is_tty() or _in_jupyter()))
    _package.addHandler(_ConsoleHandler(logging.getLevelName(cfg.level), bars))
    _package.propagate = False  # no duplicates via a root handler (e.g. Jupyter)


def _console() -> _ConsoleHandler | None:
    return next((h for h in _package.handlers if isinstance(h, _ConsoleHandler)), None)


def _stderr_is_tty() -> bool:
    try:
        return sys.stderr.isatty()
    except (AttributeError, ValueError):
        return False


def _in_jupyter() -> bool:
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and "IPKernelApp" in shell.config


@contextmanager
def capture(lines: list[str]) -> Iterator[None]:
    """Append every package record (DEBUG and up, `[HH:MM:SS] LEVEL message`) to `lines` while open."""
    handler = _ListHandler(lines)
    _package.addHandler(handler)
    try:
        yield
    finally:
        _package.removeHandler(handler)


class Progress:
    """processed/total for one long loop: a transient bar when the console has bars on,
    otherwise a `PROGRESS` INFO line about every 10%; always a final INFO count line.
    `update` is thread-safe (parallel workers may call it)."""

    def __init__(self, iterable: Iterable[T] | None, total: int | None, desc: str):
        self.iterable = iterable
        self.total = total if total is not None else len(iterable)  # type: ignore[arg-type]
        self.desc = desc
        self.n = 0
        self.postfix: dict[str, object] = {}
        self._next_line = self._step()
        self._lock = threading.Lock()
        console = _console()
        self._bar = None
        if console is not None and console.bars and self.total:
            self._bar = tqdm(total=self.total, desc=desc, leave=False, file=sys.stderr, dynamic_ncols=True)

    def _step(self) -> int:
        return max(1, -(-self.total // PROGRESS_STEPS))

    def _counts(self) -> str:
        extra = "".join(f" {k}={v}" for k, v in self.postfix.items())
        return f"{self.desc}: {self.n}/{self.total}{extra}"

    def set_postfix(self, **values: object) -> None:
        self.postfix.update(values)
        if self._bar is not None:
            self._bar.set_postfix(values, refresh=False)

    def update(self, n: int = 1) -> None:
        with self._lock:
            self.n += n
            if self._bar is not None:
                self._bar.update(n)
            if self.n >= self._next_line and self.n < self.total:
                log.info(self._counts(), extra=PROGRESS)
                self._next_line = (self.n // self._step() + 1) * self._step()

    def _close_bar(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def close(self) -> None:
        """Remove the bar, then the final INFO count line."""
        self._close_bar()
        if self.total:
            log.info(self._counts() + " done")

    def __enter__(self) -> Progress:
        return self

    def __exit__(self, exc_type, *_) -> None:
        self.close() if exc_type is None else self._close_bar()  # no "done" line after an error

    def __iter__(self) -> Iterator[T]:
        try:
            for item in self.iterable:  # type: ignore[union-attr]
                yield item
                self.update()
        except BaseException:  # error in the loop body, or the loop left early (GeneratorExit)
            self._close_bar()
            raise
        self.close()


def progress(iterable: Iterable[T] | None = None, total: int | None = None, desc: str = "") -> Progress:
    """`for x in progress(items, desc="naming"): ...`, or `with progress(total=n, desc=...) as p: p.update()`."""
    return Progress(iterable, total, desc)
