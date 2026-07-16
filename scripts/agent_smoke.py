"""Manual smoke test for the agent runtime + LangSmith tracing (CUE-21).

Run it after setting the agent and LangSmith env vars (see `.env.example` /
the values reported when this scaffold landed):

    AGENT_MODEL_PROVIDER=anthropic
    AGENT_MODEL_NAME=claude-opus-4-8
    AGENT_LANGSMITH_TRACING=true
    AGENT_LANGSMITH_PROJECT=cue-agent
    LANGSMITH_API_KEY=ls-...

    uv run python scripts/agent_smoke.py

It invokes the compiled graph once and, when a `LANGSMITH_API_KEY` is present,
queries the LangSmith API to confirm the run was recorded under the project -
proving the trace is queryable via the SDK, not merely visible in the UI.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid

from app.agent.config import agent_settings
from app.agent.graph import build_graph
from app.agent.observability import configure_tracing
from app.agent.state import AgentState

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s: %(message)s")
logger = logging.getLogger("agent_smoke")


async def main() -> None:
    """Invoke the graph once and confirm the LangSmith trace is queryable."""
    tracing_enabled = configure_tracing()

    session_id = str(uuid.uuid4())
    state: AgentState = {"session_id": session_id, "user_id": 0, "messages": []}
    graph = build_graph().compile()
    result = await graph.ainvoke(state)
    logger.info("graph output messages: %s", [m.content for m in result["messages"]])

    if not tracing_enabled:
        logger.warning("Tracing disabled - set LANGSMITH_API_KEY to record a trace.")
        return

    # LangSmith ingests runs asynchronously; give it a moment before querying.
    from langsmith import Client

    await asyncio.sleep(2.0)
    project = os.environ.get("LANGSMITH_PROJECT", agent_settings.LANGSMITH_PROJECT)
    runs = list(Client().list_runs(project_name=project, limit=1))
    if runs:
        logger.info(
            "queried LangSmith run in %r: name=%s id=%s",
            project,
            runs[0].name,
            runs[0].id,
        )
    else:
        logger.error("no runs found in project %r - check credentials/project", project)


if __name__ == "__main__":
    asyncio.run(main())
