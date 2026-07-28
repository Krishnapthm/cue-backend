"""The model seam: a node asks for a *role*, config decides everything else.

The langchain chat-model constructors validate that a provider API key is
present (they build the underlying client eagerly), so these tests inject dummy
keys. The keys are never used - no request is made - they only let construction
succeed offline.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Literal

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.agent.config import AgentSettings, ModelRole, ReasoningEffort
from app.agent.providers import get_chat_model

NODES_DIR = Path("app/agent/nodes")


@pytest.fixture(autouse=True)
def _provider_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")


def _settings(
    provider: Literal["openai", "anthropic"] = "openai", **overrides: Any
) -> AgentSettings:
    return AgentSettings(MODEL_PROVIDER=provider, **overrides)


# --- provider seam ---------------------------------------------------------


def test_openai_provider_returns_openai_chat_model() -> None:
    model = get_chat_model(ModelRole.RECIPE, _settings(MODEL_RECIPE="gpt-5.6-luna"))

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gpt-5.6-luna"


def test_openai_is_the_default_provider() -> None:
    # PRD Section 12 is settled: OpenAI unless a deployment says otherwise.
    assert AgentSettings.model_fields["MODEL_PROVIDER"].default == "openai"


def test_anthropic_provider_returns_anthropic_chat_model() -> None:
    model = get_chat_model(
        ModelRole.RECIPE, _settings("anthropic", MODEL_RECIPE="claude-opus-4-8")
    )

    assert isinstance(model, ChatAnthropic)
    assert model.model == "claude-opus-4-8"


def test_every_role_shares_the_base_chat_model_interface() -> None:
    # The graph and its nodes depend only on BaseChatModel, which is what keeps
    # a model swap free of any change to graph.py.
    for role in ModelRole:
        assert isinstance(get_chat_model(role, _settings()), BaseChatModel)


# --- per-role selection ----------------------------------------------------


def test_each_role_resolves_to_its_own_configured_model() -> None:
    settings = _settings(
        MODEL_ROUTER="router-model",
        MODEL_RECIPE="recipe-model",
        MODEL_VISION="vision-model",
    )

    assert settings.model_for(ModelRole.ROUTER).model_id == "router-model"
    assert settings.model_for(ModelRole.RECIPE).model_id == "recipe-model"
    assert settings.model_for(ModelRole.VISION).model_id == "vision-model"


@pytest.mark.parametrize(
    ("role", "env_var", "expected_attr"),
    [
        (ModelRole.ROUTER, "AGENT_MODEL_ROUTER", "model_name"),
        (ModelRole.RECIPE, "AGENT_MODEL_RECIPE", "model_name"),
        (ModelRole.VISION, "AGENT_MODEL_VISION", "model_name"),
    ],
)
def test_every_role_is_overridable_by_env_var(
    monkeypatch: pytest.MonkeyPatch,
    role: ModelRole,
    env_var: str,
    expected_attr: str,
) -> None:
    monkeypatch.setenv(env_var, "swapped-by-env")

    model = get_chat_model(role, AgentSettings())

    assert getattr(model, expected_attr) == "swapped-by-env"


def test_the_router_runs_at_the_configured_reasoning_effort() -> None:
    settings = _settings(MODEL_ROUTER_REASONING_EFFORT=ReasoningEffort.NONE)

    model = get_chat_model(ModelRole.ROUTER, settings)

    assert isinstance(model, ChatOpenAI)
    assert model.reasoning_effort == "none"


def test_reasoning_effort_is_overridable_by_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_MODEL_ROUTER_REASONING_EFFORT", "high")

    model = get_chat_model(ModelRole.ROUTER, AgentSettings())

    assert isinstance(model, ChatOpenAI)
    assert model.reasoning_effort == "high"


def test_roles_without_the_effort_dial_send_no_effort_kwarg() -> None:
    # luna does not offer reasoning effort; sending one anyway would be a
    # provider error on the first real turn.
    for role in (ModelRole.RECIPE, ModelRole.VISION):
        assert _settings().model_for(role).reasoning_effort is None
        model = get_chat_model(role, _settings())
        assert isinstance(model, ChatOpenAI)
        assert model.reasoning_effort is None


def test_an_unknown_reasoning_effort_fails_at_config_load() -> None:
    # A typo must fail loudly at startup, not silently on a live turn.
    with pytest.raises(ValueError, match="MODEL_ROUTER_REASONING_EFFORT"):
        _settings(MODEL_ROUTER_REASONING_EFFORT="exhaustive")


# --- the invariant ---------------------------------------------------------


def _string_constants(path: Path) -> set[str]:
    """Every string literal in a module, as the source actually spells them."""
    tree = ast.parse(path.read_text())
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


@pytest.mark.parametrize(
    "node_file", sorted(NODES_DIR.glob("*.py")), ids=lambda p: p.name
)
def test_no_node_hard_codes_a_model_id_or_an_effort(node_file: Path) -> None:
    # `config.py`'s docstring is the rule this enforces: model choice must
    # stay swappable via config alone. A node names a role, never a model -
    # and never an effort, which belongs on the same settings-driven path.
    model_ids = {
        str(AgentSettings.model_fields[field].default)
        for field in ("MODEL_ROUTER", "MODEL_RECIPE", "MODEL_VISION")
    }
    literals = _string_constants(node_file)

    assert not (literals & model_ids), f"{node_file.name} hard-codes a model id"
    assert "reasoning_effort" not in node_file.read_text(), (
        f"{node_file.name} sets a reasoning effort inline"
    )
