from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "tickets.xlsx"


@pytest.fixture
def fixture_path() -> Path:
    return FIXTURE


@pytest.fixture(autouse=True)
def _no_console_logging_leak():
    """`setup_logging` (CLI / notebook tests) adds a console handler; drop it after each test."""
    import logging

    package = logging.getLogger("problem_clustering")
    before, propagate = list(package.handlers), package.propagate
    yield
    package.handlers[:] = before
    package.propagate = propagate
