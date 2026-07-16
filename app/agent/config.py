from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    """LangGraph agent runtime settings.

    Model provider choice is an open decision (PRD Section 12) and must stay
    swappable via config alone - never hard-coded in the graph or its nodes.

    `LANGSMITH_API_KEY` and the other `LANGSMITH_*`/`LANGCHAIN_*` env vars the
    tracing SDK consumes are read directly by that SDK per its own convention,
    so they are deliberately not re-declared here. `app.agent.observability`
    bridges the two settings below onto those env vars at runtime.
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Selects which langchain-core BaseChatModel `providers.get_chat_model`
    # returns. Anthropic is the default until PRD Section 12 is settled.
    MODEL_PROVIDER: Literal["openai", "anthropic"] = "anthropic"
    # Required: the provider-specific model id (e.g. "claude-opus-4-8" or
    # "gpt-4o"). No default - the deployment must state it explicitly.
    MODEL_NAME: str
    # When true, runs are sent to LangSmith *if* a LANGSMITH_API_KEY is present;
    # a missing key degrades to a logged warning, never a hard failure.
    LANGSMITH_TRACING: bool = True
    LANGSMITH_PROJECT: str = "cue-agent"


@lru_cache
def get_agent_settings() -> AgentSettings:
    """Return the cached agent settings."""
    return AgentSettings()


agent_settings = get_agent_settings()
