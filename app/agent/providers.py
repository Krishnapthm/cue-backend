from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.agent.config import AgentSettings, agent_settings


def get_chat_model(settings: AgentSettings = agent_settings) -> BaseChatModel:
    """Return the configured provider's chat model.

    The model is returned behind langchain-core's `BaseChatModel` so graph
    nodes depend only on that interface and never import `openai`/`anthropic`
    (or their langchain wrappers) directly. Swapping `AGENT_MODEL_PROVIDER`
    therefore changes the model with no edit to `graph.py` or any node - the
    provider seam lives entirely here.

    Args:
        settings: Agent settings to read the provider and model id from;
            defaults to the process-wide cached settings.

    Returns:
        A ready-to-invoke chat model for the configured provider.
    """
    match settings.MODEL_PROVIDER:
        case "openai":
            return ChatOpenAI(model=settings.MODEL_NAME)
        case "anthropic":
            return ChatAnthropic(model=settings.MODEL_NAME)
