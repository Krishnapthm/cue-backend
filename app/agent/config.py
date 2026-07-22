from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    """LangGraph agent runtime settings.

    Model provider choice is an open decision (PRD Section 12) and must stay
    swappable via config alone - never hard-coded in the graph or its nodes.
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Selects which langchain-core BaseChatModel `providers.get_chat_model`
    # returns. PRD Section 12 is settled: OpenAI. Anthropic stays wired up so
    # the seam is a live two-provider choice rather than a dead branch.
    MODEL_PROVIDER: Literal["openai", "anthropic"] = "openai"
    # Required: the provider-specific model id (e.g. "gpt-5.4" or
    # "claude-opus-4-8"). No default - the deployment must state it
    # explicitly, so a model swap is always a visible config change.
    MODEL_NAME: str

    # Supabase project base URL, used to build recipe-photo object URLs (see
    # `app.agent.storage.SupabaseImageStore`). Optional so the app still
    # imports without it configured; `SupabaseImageStore.load` raises a clear
    # error if it is unset when actually invoked.
    SUPABASE_URL: str | None = None
    # Supabase Storage bucket that recipe photo uploads land in.
    RECIPE_IMAGE_BUCKET: str = "recipe-images"


@lru_cache
def get_agent_settings() -> AgentSettings:
    """Return the cached agent settings."""
    return AgentSettings()


agent_settings = get_agent_settings()
