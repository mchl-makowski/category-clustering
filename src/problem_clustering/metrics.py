"""Judge and Metrics sheets of a run workbook. Metrics = long format (`metric`, `value`), no cost estimate."""

from __future__ import annotations

import pandas as pd

from .judge import JUDGE_COLUMNS, JudgeResult
from .result import EMPTY_PROBLEM_ID, RunResult

METRIC_COLUMNS = ["metric", "value"]
USAGE_KEYS = ["llm_calls", "prompt_tokens", "completion_tokens", "embedding_calls", "embedding_tokens", "runtime_s"]


def judge_sheet(judge_result: JudgeResult | None) -> pd.DataFrame:
    if judge_result is None:
        return pd.DataFrame(columns=JUDGE_COLUMNS)
    return judge_result.checks[JUDGE_COLUMNS]


def run_metrics(run_result: RunResult, judge_result: JudgeResult | None) -> dict[str, object]:
    cat = run_result.catalogue
    is_empty = cat["problem_id"] == EMPTY_PROBLEM_ID
    sizes = cat.loc[~is_empty, "ticket_count"].astype(int)
    has = len(sizes) > 0
    m: dict[str, object] = {
        "method": run_result.method,
        "granularity": run_result.config.granularity,
        "tickets": len(run_result.tickets),
        "empty_tickets": int(cat.loc[is_empty, "ticket_count"].sum()),
        "problem_count": len(sizes),
        "one_ticket_share": float((sizes == 1).mean()) if has else None,
        "size_min": int(sizes.min()) if has else None,
        "size_p25": float(sizes.quantile(0.25)) if has else None,
        "size_median": float(sizes.median()) if has else None,
        "size_mean": float(sizes.mean()) if has else None,
        "size_p75": float(sizes.quantile(0.75)) if has else None,
        "size_max": int(sizes.max()) if has else None,
        "purity": judge_result.purity if judge_result else None,
        "split_rate": judge_result.split_rate if judge_result else None,
        "judge_mixed_checks": len(judge_result.mixed_checks) if judge_result else None,
        "judge_split_pairs": len(judge_result.split_checks) if judge_result else None,
    }
    usage = run_result.usage.as_dict()
    m.update({k: usage[k] for k in USAGE_KEYS})
    judge_usage = judge_result.usage.as_dict() if judge_result else {}
    m.update({f"judge_{k}": judge_usage.get(k) for k in USAGE_KEYS})
    return m


def metrics_sheet(run_result: RunResult, judge_result: JudgeResult | None) -> pd.DataFrame:
    return pd.DataFrame(list(run_metrics(run_result, judge_result).items()), columns=METRIC_COLUMNS)
