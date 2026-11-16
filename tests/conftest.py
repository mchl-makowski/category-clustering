from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "tickets.xlsx"


@pytest.fixture
def fixture_path() -> Path:
    return FIXTURE
