"""Service Desk Problem clustering. Public API (the one seam for CLI, notebook and tests)."""

from .config import Config, load_config
from .logs import setup_logging
from .output import write_outputs
from .result import RunResult
from .run import run_method
from .tickets import load_tickets
from .judge import JudgeResult, judge, judge_safely  # noqa: E402  (#7)
from .compare import compare  # noqa: E402  (#7)

__all__ = ["Config", "RunResult", "load_config", "load_tickets", "run_method", "write_outputs"]
__all__ += ["JudgeResult", "judge", "judge_safely", "compare"]  # #7
__all__ += ["setup_logging"]
