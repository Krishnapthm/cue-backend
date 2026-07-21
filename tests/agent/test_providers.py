"""The provider seam: `AGENT_MODEL_PROVIDER` swaps the model, nothing else.

The langchain chat-model constructors validate that a provider API key is
present (they build the underlying client eagerly), so these tests inject dummy
keys. The keys are never used - no request is made - they only let construction
succeed offline.
"""

from __future__ import annotations

from typing import Literal

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.agent.config import AgentSettings
from app.agent.providers import get_chat_model


@pytest.fixture(autouse=True)
def _provider_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")


def _settings(provider: Literal["openai", "anthropic"], model: str) -> AgentSettings:
    return AgentSettings(MODEL_PROVIDER=provider, MODEL_NAME=model)


def test_openai_provider_returns_openai_chat_model() -> None:
    model = get_chat_model(_settings("openai", "gpt-5.4"))

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gpt-5.4"


def test_openai_is_the_default_provider() -> None:
    # PRD Section 12 is settled: a deployment that sets only AGENT_MODEL_NAME
    # gets OpenAI. The model id itself stays required, so it is never implicit.
    assert AgentSettings.model_fields["MODEL_PROVIDER"].default == "openai"
    assert AgentSettings.model_fields["MODEL_NAME"].is_required()


def test_anthropic_provider_returns_anthropic_chat_model() -> None:
    model = get_chat_model(_settings("anthropic", "claude-opus-4-8"))

    assert isinstance(model, ChatAnthropic)
    assert model.model == "claude-opus-4-8"


def test_both_providers_share_the_base_chat_model_interface() -> None:
    # The graph and its nodes depend only on BaseChatModel, which is what keeps
    # the swap free of any change to graph.py.
    openai_model = get_chat_model(_settings("openai", "gpt-5.4"))
    anthropic_model = get_chat_model(_settings("anthropic", "claude-opus-4-8"))

    assert isinstance(openai_model, BaseChatModel)
    assert isinstance(anthropic_model, BaseChatModel)
