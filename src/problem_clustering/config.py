"""Run configuration: one pydantic model, one sub-model per section.

Extension: a later ticket adds its own section as a new sub-model class plus one
field on `Config` (e.g. `method_b: MethodBConfig = MethodBConfig()`). Every
section forbids unknown keys so YAML/CLI typos fail loudly.
"""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

GranularityLevel = Literal["coarse", "medium", "fine"]


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BackendConfig(Section):
    """Which model backend to use. Secrets/endpoint come from env (see settings.py)."""

    kind: Literal["azure", "fake"] = "azure"
    embedding_deployment: str = "text-embedding-3-large"
    chat_deployment: str = "gpt-4o"
    embed_batch_size: int = 64
    # chars of each text sent to the embedding model; text-embedding-3-large takes <= 8191 tokens,
    # and Polish averages well over 1 char/token, so 8000 chars stays under the limit
    embed_max_chars: int = 8000


class NamingConfig(Section):
    """Shared Polish naming step."""

    sample_size: int = 8  # Tickets shown to the LLM per Problem


class MethodAConfig(Section):
    algorithm: Literal["agglomerative", "hdbscan"] = "agglomerative"
    distance_threshold: float = 0.35  # cosine distance, average linkage
    linkage: Literal["average", "complete", "single"] = "average"
    min_cluster_size: int = 2  # hdbscan: smallest multi-Ticket Problem
    min_samples: int | None = None  # hdbscan: None = min_cluster_size


class SymptomSummaryConfig(Section):
    """Optional LLM rewrite of each Ticket into one Polish symptom + context line, grouped instead of raw text."""

    enabled: bool = False
    log_summaries: bool = False  # write each Ticket's summary into the run log


class RuntimeConfig(Section):
    """Disk cache, bounded parallelism and retry/backoff around any backend (#4)."""

    cache_dir: str | None = None  # None = no disk cache
    max_concurrency: int = 8  # parallel independent model calls (parallel_map)
    max_retries: int = 5  # retries after the first attempt on transient errors
    backoff_initial_s: float = 1.0  # wait before retry 1, then x backoff_multiplier
    backoff_multiplier: float = 2.0
    backoff_max_s: float = 60.0


class MethodBConfig(Section):
    coarse_threshold: float = 0.6  # cosine distance for the starting coarse clusters (looser than A)
    sample_size: int = 10  # Tickets shown to the LLM per verification / merge check
    split_step: float = 0.1  # a mixed cluster is re-clustered at its threshold minus this
    merge_neighbours: int = 1  # nearest-centroid neighbours LLM-checked for a merge, per Problem
    max_iterations: int = 5
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


class JudgeConfig(Section):
    """Judge: fixed model checking any run for Mixed and Split Problems."""

    deployment: str = "gpt-4o"  # fixed Judge chat deployment, independent of backend.chat_deployment
    sample_size: int = 8  # Tickets shown per Mixed check
    split_sample_size: int = 4  # Tickets shown per Problem in a Split check
    neighbours: int = 3  # nearest Problems (by centroid) checked for a Split Problem
    max_problems: int | None = None  # judge only the N largest Problems


class Config(Section):
    granularity: GranularityLevel = "medium"
    backend: BackendConfig = BackendConfig()
    naming: NamingConfig = NamingConfig()
    method_a: MethodAConfig = MethodAConfig()
    method_b: MethodBConfig = MethodBConfig()
    method_c: MethodCConfig = MethodCConfig()
    output: OutputConfig = OutputConfig()
    symptom_summary: SymptomSummaryConfig = SymptomSummaryConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    judge: JudgeConfig = JudgeConfig()


class DeploymentSettings(BaseSettings):
    """Deployment names from env (or `.env`): FOUNDRY_EMBEDDING_DEPLOYMENT, FOUNDRY_CHAT_DEPLOYMENT,
    FOUNDRY_JUDGE_DEPLOYMENT. A set variable wins over YAML; an explicit `--set` wins over env."""

    model_config = SettingsConfigDict(env_prefix="FOUNDRY_", env_file=".env", extra="ignore")

    embedding_deployment: str | None = None
    chat_deployment: str | None = None
    judge_deployment: str | None = None

    def as_overrides(self) -> dict[str, str]:
        keys = {"embedding_deployment": "backend.embedding_deployment",
                "chat_deployment": "backend.chat_deployment", "judge_deployment": "judge.deployment"}
        return {keys[f]: v for f, v in self.model_dump().items() if v}


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
    """Load Config from YAML (optional), then deployment names from env (`DeploymentSettings`),
    then dotted-key overrides (`method_a.distance_threshold=0.3`); later wins."""
    data: dict[str, Any] = {}
    if yaml_path is not None:
        data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8")) or {}
    layered = {**DeploymentSettings().as_overrides(), **_parse_overrides(overrides)}
    for key, value in layered.items():
        node = data
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    return Config.model_validate(data)
