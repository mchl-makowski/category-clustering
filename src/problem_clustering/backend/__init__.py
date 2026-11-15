"""Model backends: protocol, Azure Foundry, Fake, usage tracking."""

from ..config import BackendConfig
from .base import Backend, ChatReply, EmbeddingReply, ToolCall, Usage, embed_texts, json_schema_format
from .fake import FakeBackend, tool_reply
from .tracking import TrackedBackend, UsageCounters


def make_backend(config: BackendConfig) -> Backend:
    """Backend selected by `backend.kind` in Config."""
    if config.kind == "fake":
        return FakeBackend()
    from .azure import AzureFoundryBackend

    return AzureFoundryBackend(config)


__all__ = [
    "Backend", "ChatReply", "EmbeddingReply", "FakeBackend", "ToolCall", "TrackedBackend", "Usage",
    "UsageCounters", "embed_texts", "json_schema_format", "make_backend", "tool_reply",
]
