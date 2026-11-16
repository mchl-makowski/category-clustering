"""`compare`: the Metrics sheets of several run workbooks side by side.

Writes `yymmdd_hhmm_comparison.xlsx` with one sheet `Comparison`: one row per metric,
one column per run (named after the run workbook's file stem).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def compare(workbook_paths: list[str | Path], out_dir: str | Path) -> Path:
    if not workbook_paths:
        raise ValueError("compare needs at least one run workbook")
    columns: dict[str, pd.Series] = {}
    for path in map(Path, workbook_paths):
        metrics = pd.read_excel(path, sheet_name="Metrics")
        name, n = path.stem, 2
        while name in columns:
            name, n = f"{path.stem} ({n})", n + 1
        columns[name] = metrics.set_index("metric")["value"]
        log.info(f"compare: read {path.name} ({len(metrics)} metrics)")
    table = pd.DataFrame(columns)  # union of metrics, first-seen order
    table.index.name = "metric"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{datetime.now():%y%m%d_%H%M}_comparison"
    stem, n = base, 2
    while (out_dir / f"{stem}.xlsx").exists():
        stem, n = f"{base}_{n}", n + 1
    out = out_dir / f"{stem}.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="Comparison")
    log.info(f"compare: {len(columns)} runs x {len(table)} metrics written to {out}")
    return out
