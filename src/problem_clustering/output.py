"""Run outputs: `yymmdd_hhmm_<method>.xlsx` (Tickets, Problem Catalogue, Judge, Metrics) + `.txt` log.

Extension (#7): replace `judge_sheet` / `metrics_sheet` bodies; nothing else here
needs to change. `compare` will live in its own module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .result import RunResult

SHEETS = ["Tickets", "Problem Catalogue", "Judge", "Metrics"]


def judge_sheet(judge_result: Any | None) -> pd.DataFrame:
    from .metrics import judge_sheet as build  # #7

    return build(judge_result)


def metrics_sheet(run_result: RunResult, judge_result: Any | None) -> pd.DataFrame:
    from .metrics import metrics_sheet as build  # #7

    return build(run_result, judge_result)


def _stem(run_result: RunResult, out_dir: Path) -> str:
    base = f"{run_result.started_at:%y%m%d_%H%M}_{run_result.method}"
    stem, n = base, 2
    while (out_dir / f"{stem}.xlsx").exists() or (out_dir / f"{stem}.txt").exists():
        stem, n = f"{base}_{n}", n + 1
    return stem


def write_outputs(run_result: RunResult, judge_result: Any | None, out_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _stem(run_result, out_dir)
    xlsx, txt = out_dir / f"{stem}.xlsx", out_dir / f"{stem}.txt"

    sheets = {
        "Tickets": run_result.tickets_with_assignments(),
        "Problem Catalogue": run_result.catalogue,
        "Judge": judge_sheet(judge_result),
        "Metrics": metrics_sheet(run_result, judge_result),
    }
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        for name in SHEETS:
            sheets[name].to_excel(writer, sheet_name=name, index=False)

    txt.write_text("\n".join(run_result.log) + "\n", encoding="utf-8")
    return {"xlsx": xlsx, "txt": txt}
