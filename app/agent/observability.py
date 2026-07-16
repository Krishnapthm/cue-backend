from __future__ import annotations

import logging
import os

from app.agent.config import AgentSettings, agent_settings

logger = logging.getLogger(__name__)

# The env vars the LangSmith tracing SDK reads directly. We write them from
# AgentSettings so the rest of the app configures tracing through our typed
# settings, not by exporting LangChain-specific vars by hand.
_TRACING_ENV_VAR = "LANGSMITH_TRACING"
_PROJECT_ENV_VAR = "LANGSMITH_PROJECT"
_API_KEY_ENV_VAR = "LANGSMITH_API_KEY"


def configure_tracing(settings: AgentSettings = agent_settings) -> bool:
    """Bridge `AgentSettings` onto the env vars the LangSmith SDK consumes.

    LangGraph/LangChain enable tracing by reading `LANGSMITH_TRACING` and
    `LANGSMITH_PROJECT` from the environment at invocation time. This sets them
    from our typed settings and is idempotent, so it is safe to call before
    every graph build.

    Observability is never a hard dependency for execution: if tracing is
    requested but `LANGSMITH_API_KEY` is absent, this logs a warning, forces
    tracing off, and returns `False` so the graph still runs.

    Args:
        settings: Agent settings to read tracing config from; defaults to the
            process-wide cached settings.

    Returns:
        `True` if tracing is enabled, `False` if it is disabled or degraded.
    """
    if not settings.LANGSMITH_TRACING:
        os.environ[_TRACING_ENV_VAR] = "false"
        logger.info("LangSmith tracing disabled by configuration")
        return False

    if not os.getenv(_API_KEY_ENV_VAR):
        os.environ[_TRACING_ENV_VAR] = "false"
        logger.warning(
            "%s is not set; LangGraph will run without LangSmith tracing",
            _API_KEY_ENV_VAR,
        )
        return False

    os.environ[_TRACING_ENV_VAR] = "true"
    os.environ[_PROJECT_ENV_VAR] = settings.LANGSMITH_PROJECT
    logger.info("LangSmith tracing enabled for project %r", settings.LANGSMITH_PROJECT)
    return True
