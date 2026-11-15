"""Run configuration: one pydantic model, one sub-model per section.

Extension: a later ticket adds its own section as a new sub-model class plus one
field on `Config` (e.g. `method_b: MethodBConfig = MethodBConfig()`). Every
section forbids unknown keys so YAML/CLI typos fail loudly.
"""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict

GranularityLevel = Literal["coarse", "medium", "fine"]


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BackendConfig(Section):
    """Which model backend to use. Secrets/endpoint come from env (see settings.py)."""

    kind: Literal["azure", "fake"] = "azure"
    embedding_deployment: str = "text-embedding-3-large"
    chat_deployment: str = "gpt-4o"
    embed_batch_size: int = 64


class NamingConfig(Section):
    """Shared Polish naming step."""

    sample_size: int = 8  # Tickets shown to the LLM per Problem


class MethodAConfig(Section):
    algorithm: Literal["agglomerative"] = "agglomerative"  # "hdbscan" added by #6
    distance_threshold: float = 0.35  # cosine distance, average linkage
    linkage: Literal["average", "complete", "single"] = "average"


class MethodCConfig(Section):
    """Ticket-by-Ticket tool-calling agent with lookup and periodic consolidation."""

    max_tool_calls: int = 6  # per Ticket; then the Ticket gets a new Problem (zgadywane)
    search_k: int = 5  # default k for search_similar_tickets
    search_max_k: int = 20  # hard cap on k the agent may request
    consolidate_every: int = 250  # Tickets between consolidations (also runs at the end)
    consolidate_threshold: float = 0.5  # cosine distance between Problem centroids to be merge candidates
    consolidate_group_size: int = 15  # max Problems shown to the LLM per consolidation request
    example_tickets: int = 5  # Ticket texts returned by get_problem


class OutputConfig(Section):
    example_tickets: int = 5  # example Tickets per Problem in the catalogue


class Config(Section):
    granularity: GranularityLevel = "medium"
    backend: BackendConfig = BackendConfig()
    naming: NamingConfig = NamingConfig()
    method_a: MethodAConfig = MethodAConfig()
    method_c: MethodCConfig = MethodCConfig()
    output: OutputConfig = OutputConfig()


def _parse_overrides(overrides: list[str] | dict[str, Any] | None) -> dict[str, Any]:
    if not overrides:
        return {}
    if isinstance(overrides, dict):
        return dict(overrides)
    parsed = {}
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Override must be key=value, got {item!r}")
        parsed[key.strip()] = yaml.safe_load(value)  # "0.3" -> float, "true" -> bool
    return parsed


def load_config(
    yaml_path: str | Path | None = None,
    overrides: list[str] | dict[str, Any] | None = None,
) -> Config:
    """Load Config from YAML (optional) and apply dotted-key overrides (`method_a.distance_threshold=0.3`)."""
    data: dict[str, Any] = {}
    if yaml_path is not None:
        data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8")) or {}
    for key, value in _parse_overrides(overrides).items():
        node = data
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    return Config.model_validate(data)
