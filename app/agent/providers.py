from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.agent.config import AgentSettings, ModelRole, agent_settings


def get_chat_model(
    role: ModelRole, settings: AgentSettings = agent_settings
) -> BaseChatModel:
    """Return the chat model configured for one node role.

    The model is returned behind langchain-core's `BaseChatModel` so graph
    nodes depend only on that interface and never import `openai`/`anthropic`
    (or their langchain wrappers) directly. Swapping `AGENT_MODEL_PROVIDER`
    therefore changes the model with no edit to `graph.py` or any node - the
    provider seam lives entirely here.

    Nodes pass a *role*, never a model id: which model serves a role, and at
    what reasoning effort, is settings-driven (see `ModelRole`). A node that
    names a model - or an effort - has hard-coded a decision that must stay
    swappable by config alone.

    `reasoning_effort` is an OpenAI-only kwarg and is passed only for roles
    that configure one, so a role on a model without the dial never sends an
    argument the provider would reject.

    Args:
        role: What the calling node needs a model for.
        settings: Agent settings to read the provider and role ids from;
            defaults to the process-wide cached settings.

    Returns:
        A ready-to-invoke chat model for the configured provider.
    """
    choice = settings.model_for(role)
    match choice.provider or settings.MODEL_PROVIDER:
        case "openai":
            if choice.reasoning_effort is None:
                return ChatOpenAI(model=choice.model_id)
            return ChatOpenAI(
                model=choice.model_id,
                reasoning_effort=choice.reasoning_effort.value,
            )
        case "anthropic":
            return ChatAnthropic(model=choice.model_id)
